import os
import argparse
import numpy as np
import torch
from models.P2CNet import P2CNet
from datasets.dataloader import TestDataset
from torch.utils.data import DataLoader
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='./ckpt/P2C/P2CNet.pth', help="path to the saved checkpoint of model")
    parser.add_argument('--test_path', default='./demo', type=str, help='path to the test set')
    parser.add_argument('--bs_test', default=1, type=int, help='[test] batch size (default: 1)')
    parser.add_argument('--out_path', default='./results', type=str, help='path to the result')
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = P2CNet().cuda()
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    with torch.no_grad():
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        test_set = TestDataset(args.test_path)
        test_loader = DataLoader(test_set, batch_size=args.bs_test, shuffle=False, num_workers=8, pin_memory=True)
        for i, (raw_img, name) in enumerate(test_loader):
            raw_img = raw_img.cuda()
            out = model(raw_img)['lab_rgb']
            out = out.to(device="cpu").numpy().squeeze()
            out = np.clip(out * 255.0, 0, 255)
            save_img = Image.fromarray(np.uint8(out).transpose(1, 2, 0))
            save_img.save(os.path.join(args.out_path, str(name[0])))
            print('%d|%d' % (i + 1, len(test_set)))
