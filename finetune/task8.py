import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
import numpy as np
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
)
from resunet import ResUNet
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

import torch


def arg_parse():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--model', type=str, default='UNETR')
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=2)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--weight_decay', type=float, default=1e-5)
    parse.add_argument('--root_dir', type=str, default='/data/ydchen/EM_pretraining_data/MSD')
    parse.add_argument('--json_dir', type=str, default='/data/ydchen/EM_pretraining_data/MSD_json')
    parse.add_argument('--task_name', type=str, default='Task08_HepaticVessel') # need to change
    parse.add_argument('--save_dir', type=str, default='/data/ydchen/VLP/MODEL/NeurIPS_MSD_0418')
    parse.add_argument('--json_name', type=str, default='task8') # need to change
    parse.add_argument('--pretrained', type=bool, default=False)
    parse.add_argument('--pretrained_path', type=str, default='/data/ydchen/EM_pretraining_data/Siasiam0413/resnet50_80000.pth')
    parse.add_argument('--valid_iters', type=int, default=500)
    args = parse.parse_args()
    return args

args = arg_parse()

data_dir = os.path.join(args.root_dir, args.task_name)
split_json = "dataset.json"
patch_json = os.path.join(args.json_dir,args.json_name,'nnUNetPlans.json')

save_dir = os.path.join(args.save_dir, args.task_name + '_pretrained_' + str(args.pretrained) + '_' + os.path.basename(args.pretrained_path).split('.')[0])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(patch_json) as patch_json:
    patch_data = json.load(patch_json)
    patch_size = patch_data['configurations']['3d_fullres']['patch_size']
    patch_size = np.array(patch_size)
    patch_size = patch_size[[1,2,0]]
    patch_size = tuple(patch_size)
    spacing = patch_data['configurations']['3d_fullres']['spacing']
    spacing = np.array(spacing)
    spacing = spacing[[1,2,0]]
    spacing = tuple(spacing)

print(f'patch_size: {patch_size}, spacing: {spacing}')
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # RandCropByPosNegLabeld(
        #     keys=["image", "label"],
        #     label_key="label",
        #     spatial_size=patch_size,
        #     pos=1,
        #     neg=1,
        #     num_samples=4,
        #     image_key="image",
        #     image_threshold=0,
        # ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=patch_size,
            random_size=False,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
   
datasets = os.path.join(data_dir, split_json)
print(datasets)
datalist = load_decathlon_datalist(datasets, True, "training")
total_len = len(datalist)
trainlist = datalist[: int(total_len * 0.8)]
val_files = datalist[int(total_len * 0.8) :]

with open(datasets) as json_file:
    data = json.load(json_file)
    labels = data['labels']
    modality = data['modality']


train_ds = CacheDataset(
    data=trainlist,
    transform=train_transforms,
    cache_num=8,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = UNETR(
#     in_channels=1,
#     out_channels=14,
#     img_size=patch_size,
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)

model = ResUNet(args, input_channel=len(modality),input_size=patch_size,out_channels=len(labels)).to(device)

model = nn.DataParallel(model)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # print(f'val_inputs: {val_inputs.shape}, val_labels: {val_labels.shape}')
            val_outputs = sliding_window_inference(val_inputs, patch_size, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        # print(f'x: {x.shape}, y: {y.shape}')
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            with open(os.path.join(save_dir, "train_log.txt"), "a") as f:
                f.write("epoch %d, global_step %d, loss %f, metric %f" % (global_step // eval_num, global_step, epoch_loss, dice_val))
                f.write("\n")
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(save_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}. Current step is {}".format(dice_val_best, dice_val, global_step)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 25000
eval_num = args.valid_iters
post_label = AsDiscrete(to_onehot=len(labels))
post_pred = AsDiscrete(argmax=True, to_onehot=len(labels))
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.savefig(os.path.join(save_dir, "train.png"))

with open(os.path.join(save_dir, "train_results.txt"), "w") as f:
    f.write("dice_val_best: {}\n".format(dice_val_best))