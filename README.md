# Deep Color Compensation for Generalized Underwater Image Enhancement
Demo and Implementation of P2CNet and two-stage UIE (An free lunch for most existing UIE models without any tuning)
![P2CNet](/samples/model.png)
Our work is accepted by IEEE Transactions on Circuits and Systems for Video Technology 2023.

# Pre-Trained Weights
The compensation network and model weights are provided in the models and ckpt files respectively.

For the enhancement network and the model weights of CLUIE, please refer to the [CLUIE-Net](https://github.com/justwj/CLUIE-Net)

# DEMO：P2CNet+CLUIE-Net
We provide the enhancement demo of using P2CNet and CLUIE-Net in demo.ipynb and test.py, which are easily to be adapted with most existing enhancement models or algorthms.

# Citation

If you are interested in this work, please cite the following work:

```
@ARTICLE{10220126,
  author={Rao, Yuan and Liu, Wenjie and Li, Kunqian and Fan, Hao and Wang, Sen and Dong, Junyu},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Deep Color Compensation for Generalized Underwater Image Enhancement}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3305777}}
```
