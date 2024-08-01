from swin_unetr import SwinUNETR
import torch
import torch.nn as nn

net = SwinUNETR(img_size = (96,96,96),in_channels=1, out_channels=3, depths=(2,4,8,2),for_pretrain=True, feature_size=48)
weights = torch.load('/braindat/lab/chenyd/MODEL/0426SwinUnetr_EM/model_swinvit.pt')

new_weights = {}
for k, v in weights.items():
    new_weights[k.replace('module.', '')] = v

for name in net.state_dict().keys():
    new_name = name.replace('swinViT.', '')
    if new_name in new_weights.keys():
        # print(name)
        # load pretrained weights
        if net.state_dict()[name].shape == new_weights[new_name].shape:
            net.state_dict()[name].copy_(new_weights[new_name])
            print(f'load {name} successfully! shape: {net.state_dict()[name].shape}')
        else:
            print(f'{name} in model has shape {net.state_dict()[name].shape}, while {new_name} in pretrained weights has shape {new_weights[new_name].shape}.')