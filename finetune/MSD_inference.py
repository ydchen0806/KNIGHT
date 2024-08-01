import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
import numpy as np
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRange,
    CropForeground,
)
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    RandSpatialCropd,
    SpatialPadd
)
from resunet import ResUNet
from swin_unetr import SwinUNETR
import torch.nn as nn
import argparse
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from glob import glob
import nibabel as nib
import torch

import scipy.ndimage as ndimage



def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

def arg_parse():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--model', type=str, default='UNETR')
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=2)
    parse.add_argument('--lr', type=float, default=1e-5)
    parse.add_argument('--weight_decay', type=float, default=1e-5)
    parse.add_argument('--root_dir', type=str, default='/data/ydchen/EM_pretraining_data/MSD')
    parse.add_argument('--json_dir', type=str, default='/data/ydchen/EM_pretraining_data/MSD_json')
    parse.add_argument('--task_name', type=str, default='Task01_BrainTumour') # need to change
    parse.add_argument('--save_dir', type=str, default='/data/ydchen/VLP/MODEL/NeurIPS_MSD_0421_barlowClip_swin0502')
    parse.add_argument('--json_name', type=str, default='task1') # need to change
    parse.add_argument('--pretrained', type=bool, default=True)
    parse.add_argument('--pretrained_path', type=str, default='/data/ydchen/VLP/MODEL/NeurIPS_MSD_0418/Task01_BrainTumour_pretrained_False_resnet50_80000/best_metric_model.pth')
    parse.add_argument('--valid_iters', type=int, default=500)
    parse.add_argument('--max_iterations', type=int, default=40000)
    parse.add_argument('--backbone', type=str, default='resnet50', help='resnet50 or SwinUNETR')
    parse.add_argument('--load_model', type=bool, default=True)
    parse.add_argument('--seg_path', type=str, default='/data/ydchen/VLP/MODEL/NeurIPS_MSD_0421_barlowClip_test0501/Task01_BrainTumour_pretrained_True_testgenerate_then_optimize_pretrain_7000_iterations_encoder/best_metric_model.pth')
    args = parse.parse_args()
    return args

args = arg_parse()
print(args)
data_dir = os.path.join(args.root_dir, args.task_name)
split_json = "dataset.json"
patch_json = os.path.join(args.json_dir,args.json_name,'nnUNetPlans.json')

save_dir = os.path.join('/data/ydchen/VLP/MSD_result',args.task_name+'_results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir,'args.txt'), 'w') as f:
    f.write(str(args))

with open(patch_json) as patch_json:
    patch_data = json.load(patch_json)
    patch_size = patch_data['configurations']['3d_fullres']['patch_size']
    patch_size = np.array(patch_size)
    # patch_size = patch_size[[1,2,0]]
    patch_size = tuple(patch_size)
    spacing = patch_data['configurations']['3d_fullres']['spacing']
    spacing = np.array(spacing)
    # spacing = spacing[[1,2,0]]
    spacing = tuple(spacing)

print(f'patch_size: {patch_size}, spacing: {spacing}')

# val_transforms = Compose(
#     [
#         LoadImaged(keys=["image"], reader="NiBabelReader"),
#         EnsureChannelFirst(channel_dim=-1),
#         Orientation(axcodes="RAS"),
#         Spacing(pixdim=spacing, mode="bilinear"),
#         ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
#         CropForeground(),
#     ]
# )
val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=spacing,
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
    ]
)
   
datasets = os.path.join(data_dir, split_json)
print(datasets)
val_files = glob(os.path.join(data_dir, 'imagesTs', '*.nii.gz'))
val_files = [{'image':path} for path in val_files]

with open(datasets) as json_file:
    data = json.load(json_file)
    labels = data['labels']
    modality = data['modality']

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.backbone == 'resnet50':
    model = ResUNet(args, input_channel=len(modality),input_size=patch_size,out_channels=len(labels))
elif args.backbone == 'SwinUNETR':
    model = SwinUNETR(args, img_size = patch_size,in_channels=len(modality), out_channels=len(labels), depths=(2,4,2,2),for_pretrain=False, feature_size=48)

if args.load_model:
    weights = torch.load(args.seg_path)
    new_weights = {}
    for k, v in weights.items():
        if k.startswith('module.'):
            new_weights[k.replace('module.', '')] = v
        else:
            new_weights[k] = v
    model.load_state_dict(new_weights)
    print(f'load model from {args.seg_path}')

model = model.to(device)
model = nn.DataParallel(model)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

with torch.no_grad():
    for i, batch in tqdm(enumerate(val_loader)):
        img = batch['image']
        print(img.shape)
        val_out = sliding_window_inference(img.cuda(), patch_size, 4, model, overlap=0.5, mode="gaussian")
        val_out = torch.softmax(val_out, 1).cpu().numpy()
        val_out = np.argmax(val_out, axis=1).astype(np.uint8)[0]
        print(f'val_out.shape: {val_out.shape}')
        basename = os.path.basename(batch['image_meta_dict']['filename_or_obj'][0])
        nib.save(nib.Nifti1Image(val_out, np.eye(4)), os.path.join(save_dir, basename))
        # print(f'save to /data/ydchen/VLP/Neurips23_caption/finetune/results/test_{i}.nii.gz')
        # val_out = resample_3d(val_out, target_shape)
