from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import h5py
import math
import time
import torch
import random
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append('/braindat/lab/chenyd/code/Neurips23_caption/predict_then_optimize/pretrain')
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.augmentation import SimpleAugment
from utils.consistency_aug_perturbations import Rescale
from utils.consistency_aug_perturbations import Filp
from utils.consistency_aug_perturbations import Intensity
from utils.consistency_aug_perturbations import GaussBlur
from utils.consistency_aug_perturbations import GaussNoise
from utils.consistency_aug_perturbations import Cutout
from utils.consistency_aug_perturbations import SobelFilter
from utils.consistency_aug_perturbations import Mixup
from utils.consistency_aug_perturbations import Elastic
from utils.consistency_aug_perturbations import Artifact
from utils.consistency_aug_perturbations import Missing
from utils.consistency_aug_perturbations import BlurEnhanced
from transformers import AutoProcessor, BlipForConditionalGeneration
import nibabel as nib
import re


def reader(data_path):
	# print(data_path)
	if data_path.endswith('.gz'):
		img = nib.load(data_path)
		img = img.get_fdata()
		if img.ndim != 3:
			if img.ndim == 4:
				selected = random.randint(0, img.shape[3] - 1)
				img = img[:, :, :, selected]
			else:
				raise AttributeError('No this data type!')
		img = np.transpose(img, (2, 1, 0))
		# print(img.shape)
	elif data_path.endswith('.hdf') or data_path.endswith('.h5'):
		f = h5py.File(data_path, 'r')
		img = f['main'][:]
		if img.ndim != 3 and img.ndim != 4:
			raise AttributeError('No this data type!')
		f.close()
	else:
		raise AttributeError('No this data type!')
	# img = scaler(img)
	
	return img


class Train(Dataset):
	def __init__(self,  cfg):
		super(Train, self).__init__()
		self.cfg = cfg
		self.model_type = cfg.MODEL.model_type
		self.per_mode = cfg.DATA.per_mode

		# basic settings
		# the input size of network
		if cfg.MODEL.model_type == 'superhuman':
			self.crop_size = [18, 160, 160]
			self.net_padding = [0, 0, 0]
		elif cfg.MODEL.model_type == 'mala':
			self.crop_size = [53, 268, 268]
			self.net_padding = [14, 106, 106]  # the edge size of patch reduced by network
		elif cfg.MODEL.model_type == 'UNETR':
			self.crop_size = [32, 160, 160]
			self.net_padding = [0, 0, 0]
		elif cfg.MODEL.model_type == 'ResUNet':
			self.crop_size = [24, 224, 224]
			self.net_padding = [0,0,0]
		else:
			raise AttributeError('No this model type!')

		# the output size of network
		# for mala: [25, 56, 56]
		# for superhuman: [18, 160, 160]
		self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

		# training dataset files (h5), may contain many datasets
		if cfg.DATA.type == 'EM':
			self.folder_name = cfg.DATA.folder_name
			self.father_path = glob('/braindat/lab/chenyd/DATASET/miccai_pretrain_data/' + '*EM*')
		elif cfg.DATA.type == 'CT':
			self.folder_name = ['/braindat/lab/chenyd/DATASET/MSD/Task03_Liver', '/braindat/lab/chenyd/DATASET/MSD/Task06_Lung', \
				'/braindat/lab/chenyd/DATASET/MSD/Task05_Prostate', '/braindat/lab/chenyd/DATASET/MSD/Task08_HepaticVessel','/braindat/lab/chenyd/DATASET/MSD/Task10_Colon']
			self.father_path = []
			for i in self.folder_name:
				try:
					self.father_path += glob(os.path.join(i,'images*'))
				except:
					continue
		elif cfg.DATA.type == 'MRI':
			self.folder_name = ['/braindat/lab/chenyd/DATASET/MSD/Task01_BrainTumour','/braindat/lab/chenyd/DATASET/MSD/Task04_Hippocampus',\
				'/braindat/lab/chenyd/DATASET/MSD/Task02_Heart','/braindat/lab/chenyd/DATASET/MSD/Task07_Pancreas']
			self.father_path = []
			for i in self.folder_name:
				try:
					self.father_path += glob(os.path.join(i,'images*'))
				except:
					continue
		elif cfg.DATA.type == 'all':
			self.folder_name = ['/braindat/lab/chenyd/DATASET/MSD/Task01_BrainTumour','/braindat/lab/chenyd/DATASET/MSD/Task04_Hippocampus',\
				'/braindat/lab/chenyd/DATASET/MSD/Task02_Heart','/braindat/lab/chenyd/DATASET/MSD/Task07_Pancreas',\
				'/braindat/lab/chenyd/DATASET/MSD/Task03_Liver', '/braindat/lab/chenyd/DATASET/MSD/Task06_Lung', \
				'/braindat/lab/chenyd/DATASET/MSD/Task05_Prostate', '/braindat/lab/chenyd/DATASET/MSD/Task08_HepaticVessel','/braindat/lab/chenyd/DATASET/MSD/Task10_Colon']
			self.father_path = []
			for i in self.folder_name:
				try:
					self.father_path += glob(os.path.join(i,'images*'))
				except:
					continue
			self.folder_name = cfg.DATA.folder_name
			self.father_path += glob('/braindat/lab/chenyd/DATASET/miccai_pretrain_data/' + '*EM*')
		else:
			raise AttributeError('No this data type!')
		# split training data
		# self.train_split = cfg.DATA.train_split
		# self.unlabel_split = cfg.DATA.unlabel_split
		# print(self.father_path)
		# augmentation
		self.simple_aug = SimpleAugment()
		self.if_norm_images = cfg.DATA.if_norm_images
		self.if_scale_aug = True
		self.scale_factor = cfg.DATA.scale_factor
		self.if_filp_aug = True
		self.if_rotation_aug = False
		self.if_intensity_aug = cfg.DATA.if_intensity_aug_unlabel
		self.if_elastic_aug = cfg.DATA.if_elastic_aug_unlabel
		self.if_noise_aug = cfg.DATA.if_noise_aug_unlabel
		self.min_noise_std = cfg.DATA.min_noise_std
		self.max_noise_std = cfg.DATA.max_noise_std
		self.if_mask_aug = cfg.DATA.if_mask_aug_unlabel
		self.if_blur_aug = cfg.DATA.if_blur_aug_unlabel
		self.min_kernel_size = cfg.DATA.min_kernel_size
		self.max_kernel_size = cfg.DATA.max_kernel_size
		self.min_sigma = cfg.DATA.min_sigma
		self.max_sigma = cfg.DATA.max_sigma
		self.if_sobel_aug = cfg.DATA.if_sobel_aug_unlabel
		self.if_mixup_aug = cfg.DATA.if_mixup_aug_unlabel
		self.if_misalign_aug = cfg.DATA.if_misalign_aug_unlabel
		self.if_artifact_aug = cfg.DATA.if_artifact_aug_unlabel
		self.if_missing_aug = cfg.DATA.if_missing_aug_unlabel
		self.if_blurenhanced_aug = cfg.DATA.if_blurenhanced_aug_unlabel

		self.train_datasets = []
		for i in self.father_path:
			self.train_datasets += glob(i + '/*.hdf')
			self.train_datasets += glob(i + '/*.gz')
		self.data_len = len(self.train_datasets)
		self.unlabel_split_rate = cfg.DATA.unlabel_split_rate
		# load dataset

		# padding by 'reflect' mode for mala network
		# if cfg.MODEL.model_type == 'mala':
		# 	for k in range(len(self.dataset)):
		# 		self.dataset[k] = np.pad(self.dataset[k], ((self.net_padding[0], self.net_padding[0]), \
		# 												   (self.net_padding[1], self.net_padding[1]), \
		# 												   (self.net_padding[2], self.net_padding[2])), mode='reflect')

		# # the training dataset size
		# self.raw_data_shape = list(self.dataset[0].shape)

		# padding for augmentation
		self.sub_padding = [0, 80, 80]  # for rescale
		# self.sub_padding = [0, 0, 0]
		self.crop_from_origin = [self.crop_size[i] + 2*self.sub_padding[i] for i in range(len(self.sub_padding))]
		# print('crop_from_origin', self.crop_from_origin)
		if not cfg.MODEL.return_dict:
			self.processor = AutoProcessor.from_pretrained('/braindat/lab/chenyd/MODEL/Neurips23/model/processor_0')
			self.text_generator = BlipForConditionalGeneration.from_pretrained('/braindat/lab/chenyd/MODEL/Neurips23/model/model_12').cuda()
		# perturbations
		self.perturbations_init()
		

	def __getitem__(self, index):
		# random select one dataset if contain many datasets
		k = random.randint(0, len(self.train_datasets)-1)
		FLAG = True
		while FLAG:
			try:
				data = reader(self.train_datasets[k])
			except:
				k = random.randint(0, len(self.train_datasets)-1)
				data = reader(self.train_datasets[k])
			if not all(data.shape[i] >= self.crop_from_origin[i] for i in range(len(self.crop_from_origin))):
				# padding
				# print(data.shape)
				pad_width = []
				for i in range(len(data.shape)):
					width = self.crop_from_origin[i] - data.shape[i]
					if width > 0:
						pad_width.append((width // 2, width - width // 2))
					else:
						pad_width.append((0, 0))
				data = np.pad(data, pad_width=pad_width, mode='reflect')
			if all(data.shape[i] >= self.crop_from_origin[i] for i in range(len(self.crop_from_origin))):
				FLAG = False
		# print(f'name is {self.train_datasets[k]}')
		self.k = k
		self.unlabel_split = int(data.shape[0] * self.unlabel_split_rate)
		
		if data.shape[0] > self.unlabel_split:
			# The default is to use the bottom 100 sections of volumes as unlabeled data for simplicity
			# However, you can use any other sections (except from labeled sections) as unlabeled data according to your needs
			data = data[:self.unlabel_split]
		# self.dataset.append(data)

		if self.cfg.MODEL.model_type == 'mala':
			data = np.pad(data, ((self.net_padding[0], self.net_padding[0]), \
														   (self.net_padding[1], self.net_padding[1]), \
														   (self.net_padding[2], self.net_padding[2])), mode='reflect')

		# the training dataset size
		self.raw_data_shape = list(data.shape)
		used_data = data

		# random select one sub-volume
		random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
		random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
		random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
		imgs = used_data[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()

		# do augmentation
		imgs = imgs.astype(np.float32) / 255.0
		
		[imgs] = self.simple_aug([imgs])
		
		# gt_imgs = imgs[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
		imgs1, _, _, _ = self.apply_perturbations(imgs, None, mode=self.per_mode)
		
		# imgs2, _, _, _ = self.apply_perturbations(imgs, None, mode=self.per_mode)
		imgs1 = self.scaler(imgs1)
		if self.cfg.MODEL.return_dict:
			caption1 = self.generate_caption(imgs1)

			# BRATS_502, a slice of the ct volume corresponding to the ct image in taken with the use of the artificial surface.
			# BRATS_502, a transverse slice of the fractional anisotropy map of the volume of interest in the brain of one subject.
			# extend dimension
			imgs1 = imgs1[np.newaxis, ...]
			# imgs2 = imgs2[np.newaxis, ...]
			# gt_imgs = gt_imgs[np.newaxis, ...]
			imgs1 = np.ascontiguousarray(imgs1, dtype=np.float32)
			
			dict1 = {'imgs': imgs1, 'caption': caption1}
			torch.cuda.empty_cache()
			# imgs2 = np.ascontiguousarray(imgs2, dtype=np.float32)
			# gt_imgs = np.ascontiguousarray(gt_imgs, dtype=np.float32)
			return dict1 #, imgs2, gt_imgs
		elif self.cfg.MODEL.match_name:
			self.data_path = self.train_datasets[self.k]
			base_name = os.path.basename(self.train_datasets[self.k]).split('.')[0]
			if self.data_path.endswith('.nii.gz'):
				data_name = self.data_path.split('/')[-3]
			else:
				data_name = self.data_path.split('/')[-2]
			caption_path = os.path.join(self.cfg.TRAIN.caption_path, data_name, base_name + '.csv')
			try:
				temp_data = pd.read_csv(caption_path)
				location_list = convert_location(temp_data['location'].values)
				caption = temp_data.iloc[find_nearest_neighbor(location_list, [random_z, random_y, random_x])]['description']
			except:
				caption = base_name + ', No caption available!'

			imgs1 = imgs1[np.newaxis, ...]
			# imgs2 = imgs2[np.newaxis, ...]
			# gt_imgs = gt_imgs[np.newaxis, ...]
			imgs1 = np.ascontiguousarray(imgs1, dtype=np.float32)
			# imgs2 = np.ascontiguousarray(imgs2, dtype=np.float32)
			# gt_imgs = np.ascontiguousarray(gt_imgs, dtype=np.float32)
			
			return imgs1, caption


		else:
			imgs1 = imgs1[np.newaxis, ...]
			# imgs2 = imgs2[np.newaxis, ...]
			# gt_imgs = gt_imgs[np.newaxis, ...]
			imgs1 = np.ascontiguousarray(imgs1, dtype=np.float32)
			base_name = os.path.basename(self.train_datasets[self.k])
			base_name = str(base_name.split('.')[0])
			
			return imgs1, base_name 

	def convert_location(self, location):
		if isinstance(location, str):    
			location = eval(location)
		if not isinstance(location, list):
			location = list(location)
		# location = [int(i) for i in location]
		return location

	def find_nearest_neighbor(self, coordinates, coord_list):
		min_distance_squared = float('inf')
		# nearest_neighbor = None
		nearest_coord = None

		for k,coord in enumerate(coord_list):
			distance_squared = sum((a - b) ** 2 for a, b in zip(coordinates, coord))
			if distance_squared < min_distance_squared:
				min_distance_squared = distance_squared
				# nearest_neighbor = coord
				nearest_coord = k

		return nearest_coord

	def scaler(self, img):
		# 归一化到0-1
		# img = img.astype(np.float32)
		img = (img - np.min(img)) / (np.max(img) - np.min(img))
		return img

	def text_filter(self, text):
		pattern = re.compile(r'^figure\s\d+\s:\s|^fig\.\s\d+\.\s|^figure\s\d+\.\s|^fig\.\s\d+\s|figure\s\d+\s')
		filtered_text = re.sub(pattern, '', text)
		base_name = os.path.basename(self.train_datasets[self.k])
		base_name = base_name.split('.')[0]
		filtered_text = base_name + ', ' + filtered_text
		return filtered_text

	def generate_caption(self, tempimg):
		# print(tempimg)
		# print(tempimg.shape)
		x, _, _= tempimg.shape
		z_select = random.randint(0, x-1)
		imgs1_png = Image.fromarray(tempimg[z_select, :, :] * 255)
		imgs1_inputs = self.processor(imgs1_png, return_tensors="pt")
		pixel_values = imgs1_inputs['pixel_values'].cuda()
		generated_ids = self.text_generator.generate(pixel_values=pixel_values, max_length=50)
		generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		generated_caption = self.text_filter(generated_caption)
		return generated_caption

	def perturbations_init(self):
		self.per_rescale = Rescale(scale_factor=self.scale_factor, det_shape=self.crop_size)
		self.per_flip = Filp()
		self.per_intensity = Intensity()
		self.per_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
		self.per_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size, min_sigma=self.min_sigma, max_sigma=self.max_sigma)
		self.per_cutout = Cutout(model_type=self.model_type)
		self.per_sobel = SobelFilter(if_mean=True)
		self.per_mixup = Mixup(min_alpha=0.1, max_alpha=0.4)
		self.per_misalign = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 0, 0], prob_slip=0.2, prob_shift=0.2, max_misalign=17, padding=20)
		self.per_elastic = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 2, 2], padding=20)
		self.per_artifact = Artifact(min_sec=1, max_sec=5)
		self.per_missing = Missing(miss_fully_ratio=0.2, miss_part_ratio=0.5)
		self.per_blurenhanced = BlurEnhanced(blur_fully_ratio=0.5, blur_part_ratio=0.7)

	def apply_perturbations(self, data, auxi=None, mode=1):
		all_pers = [self.if_scale_aug, self.if_filp_aug, self.if_rotation_aug, self.if_intensity_aug, \
					self.if_noise_aug, self.if_blur_aug, self.if_mask_aug, self.if_sobel_aug, \
					self.if_mixup_aug, self.if_misalign_aug, self.if_elastic_aug, self.if_artifact_aug, \
					self.if_missing_aug, self.if_blurenhanced_aug]
		if mode == 1:
			# select used perturbations
			used_pers = []
			for k, value in enumerate(all_pers):
				if value:
					used_pers.append(k)
			# select which one perturbation to use
			if len(used_pers) == 0:
				# do nothing
				# must crop
				data = data[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				scale_size = data.shape[-1]
				rule = np.asarray([0,0,0,0], dtype=np.int32)
				rotnum = 0
				return data, scale_size, rule, rotnum
			elif len(used_pers) == 1:
				# No choise if only one perturbation can be used
				rand_per = used_pers[0]
			else:
				rand_per = random.choice(used_pers)
			# do augmentation
			# resize
			if rand_per == 0:
				data, scale_size = self.per_rescale(data)
			else:
				data = data[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				scale_size = data.shape[-1]
			# flip
			if rand_per == 1:
				data, rule = self.per_flip(data)
			else:
				rule = np.asarray([0,0,0,0], dtype=np.int32)
			# rotation
			if rand_per == 2:
				rotnum = random.randint(0, 3)
				data = np.rot90(data, k=rotnum, axes=(1,2))
			else:
				rotnum = 0
			# intensity
			if rand_per == 3:
				data = self.per_intensity(data)
			# noise
			if rand_per == 4:
				data = self.per_gaussnoise(data)
			# blur
			if rand_per == 5:
				data = self.per_gaussblur(data)
			# mask or cutout
			if rand_per == 6:
				data = self.per_cutout(data)
			# sobel
			if rand_per == 7:
				data = self.per_sobel(data)
			# mixup
			if rand_per == 8 and auxi is not None:
				data = self.per_mixup(data, auxi)
			# misalign
			if rand_per == 9:
				data = self.per_misalign(data)
			# elastic
			if rand_per == 10:
				data = self.per_elastic(data)
			# artifact
			if rand_per == 11:
				data = self.per_artifact(data)
			# missing section
			if rand_per == 12:
				data = self.per_missing(data)
			# blur enhanced
			if rand_per == 13:
				data = self.per_blurenhanced(data)
		else:
			raise NotImplementedError
		return data, scale_size, rule, rotnum

	def __len__(self):
		return int(10000)


# class Provider(object):
# 	def __init__(self, stage, cfg):
# 			#patch_size, batch_size, num_workers, is_cuda=True):
# 		self.stage = stage
# 		if self.stage == 'train':
# 			self.data = Train(cfg)
# 			self.batch_size = cfg.TRAIN.batch_size
# 			self.num_workers = cfg.TRAIN.num_workers
# 		elif self.stage == 'valid':
# 			# return valid(folder_name, kwargs['data_list'])
# 			pass
# 		else:
# 			raise AttributeError('Stage must be train/valid')
# 		self.is_cuda = cfg.TRAIN.if_cuda
# 		self.data_iter = None
# 		self.iteration = 0
# 		self.epoch = 1
	
# 	def __len__(self):
# 		return len(self.data // self.batch_size)
	
# 	def build(self):
# 		if self.stage == 'train':
# 			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
#                                              shuffle=False, drop_last=False, pin_memory=True))
# 		else:
# 			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
#                                              shuffle=False, drop_last=False, pin_memory=True))
	
# 	def next(self):
# 		if self.data_iter is None:
# 			self.build()
# 		try:
# 			batch = self.data_iter.next()
# 			self.iteration += 1
# 			if self.is_cuda:
# 				batch = batch.cuda()
# 				# batch[0] = batch[0].cuda()
# 				# batch[1] = batch[1].cuda()
# 				# batch[2] = batch[2].cuda()
# 			return batch
# 		except StopIteration:
# 			self.epoch += 1
# 			self.build()
# 			self.iteration += 1
# 			batch = self.data_iter.next()
# 			if self.is_cuda:
# 				batch = batch.cuda()
# 				# batch[0] = batch[0].cuda()
# 				# batch[1] = batch[1].cuda()
# 				# batch[2] = batch[2].cuda()
# 			return batch

def show(img3d):
	# only used for image with shape [18, 160, 160]
	num = img3d.shape[0]
	column = 5
	row = math.ceil(num / float(column))
	size = img3d.shape[1]
	img_all = np.zeros((size*row, size*column), dtype=np.uint8)
	for i in range(row):
		for j in range(column):
			index = i*column + j
			if index >= num:
				img = np.zeros_like(img3d[0], dtype=np.uint8)
			else:
				img = (img3d[index] * 255).astype(np.uint8)
			img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
	return img_all

class Provider(object):
    def __init__(self, dataset, batch_size, num_workers):
        self.data = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers 
        # self.is_cuda = is_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1
    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                            shuffle=False, drop_last=False, pin_memory=True))
		
	
    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            # if self.is_cuda:
            #     batch = batch.cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            # if self.is_cuda:
            #     batch = batch.cuda()
            return batch

def init_dist(rank, backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, rank=rank, world_size=num_gpus, **kwargs)



if __name__ == '__main__':
	import yaml
	from attrdict import AttrDict
	from utils.show import show_one
	""""""
	seed = 555
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# init_dist(rank)
	torch.cuda.empty_cache()
	cfg_file = 'pretraining_all.yaml'
	with open(os.path.join('/braindat/lab/chenyd/code/Neurips23_caption/predict_then_optimize/config',cfg_file), 'r') as f:
		cfg = AttrDict( yaml.safe_load(f) )
	
	out_path = os.path.join('/braindat/lab/chenyd/code/Neurips23_caption/predict_then_optimize', 'data_temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	# train_provider = Provider(data, 'train', cfg)
	# temp_data = train_provider.next()
	# print(temp_data['imgs'].shape)
	print(data.data_len)
	t = time.time()
	for i in range(0, 20):
		t1 = time.time()
		tmp_data1 = iter(data).__next__()
		
		img_data1 = np.squeeze(tmp_data1['imgs'])
		# tmp_data2 = np.squeeze(tmp_data2)
		# gt = np.squeeze(gt)
		print(f'img_{str(i)} shape: ', img_data1.shape,'generate caption is : ', tmp_data1['caption'], img_data1.max(), img_data1.min())
		if cfg.MODEL.model_type == 'mala':
			img_data1 = img_data1[14:-14,106:-106,106:-106]
			# tmp_data2 = tmp_data2[14:-14,106:-106,106:-106]
			# gt = gt[14:-14,106:-106,106:-106]

		print('single cost time: ', time.time()-t1)
		

		img_data1 = show_one(img_data1)
		# img_data2 = show_one(tmp_data2)
		# img_affs = show_one(gt)
		im_cat = np.concatenate([img_data1], axis=1)

		Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
	print(time.time() - t)