import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
import os
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torchvision
import torch
from dataloader.data_provider_pretraining import Train
from torch.utils.data.dataloader import DataLoader
import yaml
import sys
from utils.utils_trainer import trainer_wBert
from utils.utils_builder import *
from attrdict import AttrDict
import torch.multiprocessing as mp

# import wandb

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, rank=rank, world_size=num_gpus, **kwargs)

def ddp_main():
    device = torch.device('cuda')
    init_dist()
    torch.cuda.empty_cache()
    # rank = dist.get_rank()
    
    # print(f"Start running basic DDP example on rank {rank}.")
    # device_id = rank % torch.cuda.device_count()
    # set up
    # cfg = yaml.load(open("/braindat/lab/chenyd/code/Neurips23_caption/predict_then_optimize/config/seg_3d.yaml", "r"), Loader=yaml.FullLoader)
    
    # print(cfg)
    with open("/braindat/lab/chenyd/code/Neurips23_caption/predict_then_optimize/config/pretraining_all.yaml", "r") as f:
        cfg = AttrDict(yaml.safe_load(f))
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # loading data path

    # define image-text dataset
    train_dataset = Train(cfg)
    # building model part
    # --------------------
    
    model = ResNet_CXRBert()

    '''
    you can freeze bert from last layer to first layer.
    set num of layer in config.yaml
    default is freeze 9 layers
    '''
    if cfg['TRAIN']['free_layers'] is not None:
        for layer_idx in range(int(cfg['network']['free_layers'])):
            for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    model = DDP(model, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
    # --------------------

    # choose optimizer (no LARS, AdamW with small batch)
    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **cfg['optimizer']['params'],
        betas=(0.9, 0.999)
    )
    # ---------xw-----------
    trainer = trainer_wBert(model=model,
                            optimizer=optimizer,
                            device=device,
                            model_name=cfg['wandb_name'],
                            **cfg['trainer'])
    # --------------------
    # if rank == 0:
    #     wandb.init(project="DiagoNet", entity="cl522", name=config['wandb_name'], group="DDP")

    # --------------------
    # I_T_P_trainer
    trainer.train_w_TextEmb(train_dataset)

if __name__ == '__main__':
    ddp_main()
