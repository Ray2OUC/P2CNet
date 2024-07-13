import os, socket, random, argparse, torch, cv2, kornia
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.P2CNet import P2CNet
import torch.nn as nn
import torch.nn.functional as F
import warnings
import time

# others: definition of perception loss
class PerceptionLoss_vgg19(nn.Module):
    def __init__(self):
        super(PerceptionLoss_vgg19, self).__init__()
        vgg = models.vgg19(pretrained=True)
        # conv2_2:7  conv3_2:12  conv4_2:21  conv5_2:30
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.SmoothL1Loss()
        self.Norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, out_images, target_images):
        out_images = self.Norm(out_images)
        target_images = self.Norm(target_images)
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss

# others: definition of train dataset
# !!!Important, In our experiments, we pre-resize the train raw image and label's resolution
# to 320*320 by bicubic interpolation, Please Modify Your Own dataset
class SUIMDataset(Dataset):
    def __init__(self, data_path='/root/megadepth/suim/bicubic/', imsz=256):
        self.root_path_ = data_path
        self.data_path, self.label_path = self.random_sample()
        self.imsz = imsz
        self.size = len(self.data_path)

    def random_sample(self):
        real_path_raw = self.root_path_ + 'train/raw/'
        real_path_ref = self.root_path_ + 'train/reference/'

        img_names = os.listdir(real_path_raw)
        random.shuffle(img_names)
        img_names = img_names[:2000]
        real_data_path = [real_path_raw + p for p in img_names]
        real_label_path = [real_path_ref + p for p in img_names]

        return real_data_path, real_label_path


    def __getitem__(self, item):
        img = cv2.imread(self.data_path[item])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_label = cv2.imread(self.label_path[item])
        rgb_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)

        # random cropping
        h, w = rgb.shape[:2]
        th = self.imsz
        i = random.randint(0, h - th)
        j = random.randint(0, w - th)
        rgb = rgb[i:i + th, j:j + th]
        rgb_label = rgb_label[i:i + th, j:j + th]

        # random rotation
        times = random.randint(0, 3)
        rgb = np.rot90(rgb, k=times)
        rgb_label = np.rot90(rgb_label, k=times)

        if np.random.random() < 0.25:
            rgb = cv2.flip(rgb, 1)
            rgb_label = cv2.flip(rgb_label, 1)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        l = lab[:1, :, :] / 100.  # 0~1
        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        lab_label = kornia.color.rgb_to_lab(rgb_label)
        ab_label = lab_label[1:, :, :] / 127.  # -1~1

        return {'input': lab, 'label_ab': ab_label, 'label_rgb': rgb_label}

    def __len__(self):
        return self.size

# others: definition of test dataset
# !!!Important, In our experiments, we pre-resize the test raw image and label's resolution
# to 256*256 by bicubic interpolation, Please Modify Your Own dataset
class Suim_Valpair(Dataset):
    def __init__(self, data_path='/root/megadepth/uie/bicubic/'):
        self.root_path_ = data_path
        self.data_path, self.label_path = self.full_sample()
        self.size = len(self.data_path)

    def full_sample(self):
        path_raw = self.root_path_ + 'test/raw/'
        path_ref = self.root_path_ + 'test/reference/'

        img_names = os.listdir(path_raw)
        data_path = [path_raw + p for p in img_names]
        label_path = [path_ref + p for p in img_names]

        return data_path, label_path

    def __getitem__(self, item):
        img = cv2.imread(self.data_path[item])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_label = cv2.imread(self.label_path[item])
        rgb_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        l = lab[:1, :, :] / 100.  # 0~1
        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        lab_label = kornia.color.rgb_to_lab(rgb_label)
        ab_label = lab_label[1:, :, :] / 127.  # -1~1

        return {'input': lab, 'label_ab': ab_label, 'label_rgb': rgb_label}

    def __len__(self):
        return self.size

# Main: Training Main Loop Function
# Please ensure your workspace path existed.
warnings.filterwarnings('ignore')

# 1. hyper parameters for training
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")  # 0.0003
parser.add_argument("--lambda_full", type=float, default=10., help="full label supervision weight")
parser.add_argument("--lambda_self", type=float, default=0.01, help="self-supervision weight")
parser.add_argument("--workspace", type=str, default="/root/uie/experiment/")
args = parser.parse_args()

torch.manual_seed(ord('c') + 137)
random.seed(ord('c') + 137)
np.random.seed(ord('c') + 137)

num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate = args.lr
model_dir = args.workspace + 'weights/'
validate_dir = args.workspace + 'validates/'
weight_full = args.lambda_full
weight_self = args.lambda_self
os.makedirs(model_dir, exist_ok=True)
os.makedirs(validate_dir, exist_ok=True)


log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs',
                           datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname())
os.makedirs(log_dir)
logger = SummaryWriter(log_dir)

# 2. network definition and prior preparation
network = P2CNet()
if torch.cuda.device_count() > 1:
    print('Using {} GPU(s).'.format(torch.cuda.device_count()))
    network = torch.nn.DataParallel(network)
network = network.cuda()

optimizer = torch.optim.AdamW(network.parameters(), lr=lr_rate)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=1e-5)


# 4. loss function definition
loss_mse = nn.MSELoss().cuda()
loss_cont = PerceptionLoss_vgg19().cuda()

# 4. training framework
val_dataloader = DataLoader(Suim_Valpair(), batch_size=8, shuffle=False, num_workers=0)

iters = 0
for epoch in range(num_epochs):

    # training
    dataloader = DataLoader(SUIMDataset(), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    network.train()
    for i, batch in enumerate(dataloader):

        # training inputs
        lab = batch['input'].cuda()  # lab 0~1 & -1~1
        label_rgb = batch['label_rgb'].cuda()  # rgb 0~1
        label_ab = batch['label_ab'].cuda()  # ab -1~1

        # inference
        optimizer.zero_grad()
        output = network(lab)
        pred_ab0 = output['ab_pred0']

        pred_ab1 = output['ab_pred1']
        h1, w1 = pred_ab1.shape[2:]
        label_ab1 = F.interpolate(label_ab, size=(h1, w1), mode='bilinear', align_corners=True)

        pred_ab2 = output['ab_pred2']
        h2, w2 = pred_ab2.shape[2:]
        label_ab2 = F.interpolate(label_ab, size=(h2, w2), mode='bilinear', align_corners=True)

        pred_ab3 = output['ab_pred3']
        h3, w3 = pred_ab3.shape[2:]
        label_ab3 = F.interpolate(label_ab, size=(h3, w3), mode='bilinear', align_corners=True)

        pred_rgb = output['lab_rgb']

        # loss computation, backward
        # loss1: multi-scale ab loss
        L_loss = (0.8**0) * loss_mse(pred_ab0, label_ab) + \
                 (0.8**1) * loss_mse(pred_ab1, label_ab1) + \
                 (0.8**2) * loss_mse(pred_ab2, label_ab2) + \
                 (0.8**3) * loss_mse(pred_ab3, label_ab3)
        # loss2: perceptual loss
        P_loss = loss_cont(pred_rgb, label_rgb)
        # total loss
        loss = weight_full * L_loss + weight_self * P_loss

        loss.backward()
        optimizer.step()

        if iters % 20 == 0:
            logger.add_scalar('Train/Loss_L1_rgb', L_loss.item(), iters)
            logger.add_scalar('Train/Loss_cont', P_loss.item(), iters)
        iters += 1

        if i % 50 == 0:
            print("\r[Epoch %d/%d: batch %d/%d] [Loss_L: %.3f, Loss_P: %.3f, Loss: %.3f]"
                             % (epoch, num_epochs, i, len(dataloader), L_loss.item(),
                                P_loss.item(), loss.item()))

    scheduler.step()

    # validating
    if epoch % 5 == 0:
        runtime = []
        print("------------------Validating on SUIM -------------------")
        network.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_dataloader)):
                lab = batch['input'].cuda()
                label_rgb = batch['label_rgb'].cpu()

                start = time.perf_counter()
                output = network(lab)
                end = time.perf_counter()
                runtime.append(end-start)
                lab_rgb = output['lab_rgb'].cpu()

                # saving first 4 recovered images
                if i == 0:
                    if epoch == 0:
                        save_image(label_rgb, validate_dir + "epoch_%s_label.png" % (epoch), nrow=4)
                    save_image(lab_rgb, validate_dir + "epoch_%s_lab.png" % (epoch), nrow=4)


        runtime = np.mean(runtime)

        print("\r[Epoch %d/%d] [runtime: %.5f]" % (epoch, num_epochs, runtime))

        ## Save model checkpoints for each epoch
        torch.save(network.state_dict(), model_dir+"epoch_%d.pth" % (epoch))
logger.close()

