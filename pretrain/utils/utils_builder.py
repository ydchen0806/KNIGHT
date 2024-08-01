from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.models import resnet as torch_resnet
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from utils_resnet3d import *
import os
from swin_unetr import SwinUNETR

# raw resnet with cxrbert-genereal
import torch
import torch.nn as nn


class ResNet_CXRBert(torch.nn.Module):
    def __init__(self):
        super(ResNet_CXRBert, self).__init__()
        net = SwinUNETR(img_size = (32,160,160),in_channels=1, out_channels=1, depths=(2,4,2,2),for_pretrain=True, feature_size=48)
        self.encoder = net

        self.proj_v = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))

        self.proj_t = nn.Sequential(
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))

        # url = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        url = 'dmis-lab/biobert-v1.1'
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')

    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=128,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output

    def forward(self, img1, img2, input_ids, attention_mask):
      
        img_emb1 = self.encoder(img1)
        # reshape to (b, 2048)
        img_emb1 = img_emb1.view(img_emb1.shape[0], img_emb1.shape[1])

        img_emb2 = self.encoder(img2)
        # reshape to (b, 2048)
        img_emb2 = img_emb2.view(img_emb2.shape[0], img_emb2.shape[1])

        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state

        # project to 512 dim
        proj_img_emb1 = self.proj_v(img_emb1)
        proj_img_emb2 = self.proj_v(img_emb2)
        proj_text_emb = self.proj_t(text_emb[:, 0].contiguous())

        return {'img_emb1': img_emb1,
                'img_emb2': img_emb2,
                'proj_img_emb1': proj_img_emb1,
                'proj_img_emb2': proj_img_emb2,
                'proj_text_emb': proj_text_emb}

# simple projection head
class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = ResNet_CXRBert()
    print(model)
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    torch.cuda.empty_cache()