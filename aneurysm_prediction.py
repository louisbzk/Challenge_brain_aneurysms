#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import sys
import scipy.ndimage as scim
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path


# In[7]:


##########################################
# ENTER DATASET DIRECTORY PATH HERE ######
##########################################
DATASET_PATH = 'test_dataset'

##########################################
# ENTER PREDICTIONS DIRECTORY PATH HERE ##
##########################################
PREDICTIONS_PATH = 'predictions'

# path to model
MODEL_PATH = 'models/DiceBCE/DiceBCE_w09_checkpoint.pth'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# In[4]:


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        # x = self.conv3d(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 4
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                # print(in_channels, feat_map_channels)
                self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                self.module_dict['conv_{}_{}'.format(depth, i)] = self.conv_block
                in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2

            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict['max_pooling_{}'.format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith('conv'):
                x = op(x)
                # print(k, x.shape)
                if k.endswith('1'):
                    down_sampling_features.append(x)
            elif k.startswith('max_pooling'):
                x = op(x)
                # print(k, x.shape)

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 4
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            # print(depth)
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            # print(feat_map_channels * 4)
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict['deconv_{}'.format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict['conv_{}_{}'.format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict['conv_{}_{}'.format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict['final_conv'] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for k, op in self.module_dict.items():
            if k.startswith('deconv'):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith('conv'):
                x = op(x)
            else:
                x = op(x)
        return x


class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation='sigmoid'):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == 'sigmoid':
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        # print("Final output shape: ", x.shape)
        return x


# In[19]:


def load_data(path: str) -> Tuple[List, List]:
    data = []
    indices_to_path = []

    for root, dirs, filenames in os.walk(path):
        for file in filenames:
            f_path = os.path.join(root, file)
            indices_to_path.append(f_path)
            data.append(h5py.File(f_path))

    return data, indices_to_path


def get_raws(data: List[h5py.File]) -> np.ndarray:
    raws = np.zeros(shape=(len(data), *data[0]['raw'].shape), dtype=np.uint8)
    for i in range(len(raws)):
        raws[i] = data[i]['raw']

    return raws


def load_as_images(raw_3d: np.ndarray):
    """
    From a given raw of shape (channels, width, height), return arrays of Image of shape (channels)

    :param raw_3d: 3D scan
    :return: array of images (the 2D slices)
    """
    n_img, channels, width, height = raw_3d.shape
    raws_img = np.empty(shape=(n_img, channels), dtype=object)
    for i in range(n_img):
        for ch in range(channels):
            raws_img[i][ch] = Image.fromarray(raw_3d[i][ch], mode='L')

    return raws_img


def img_to_numpy(raws_img):
    np_imgs = np.empty(shape=(len(raws_img), len(raws_img[0]), raws_img[0]
                       [0].size[0], raws_img[0][0].size[0]), dtype=np.float32)
    for i in range(len(raws_img)):
        for ch in range(len(raws_img[0])):
            np_imgs[i][ch] = np.asarray(raws_img[i][ch], dtype=np.float32)

    return np_imgs


def crop(raws_img: np.ndarray, box_size: int):
    """
    Crop (inplace) on an area of the image, centered at the middle

    :param raws_img: the raw images
    :param box_size: size (in pixels) of the square box centered at the middle, which defines the crop area
    """
    w, h = raws_img[0, 0].size  # images are square, w == h
    if box_size > w:
        raise ValueError(f'Received zoom box size {box_size}, which is bigger than image size ({w})')

    if box_size % 2 == 0:
        _box_size = box_size + 1
    else:
        _box_size = box_size

    img_center = (w // 2, h // 2)

    # crop the image, then resize (zoom) to original size (w, h)
    # define crop box
    x_left, x_right = img_center[0] - box_size // 2, img_center[0] + box_size // 2
    y_top, y_bottom = img_center[1] - box_size // 2, img_center[1] + box_size // 2

    for i in range(len(raws_img)):
        for channel in range(len(raws_img[0])):
            raws_img[i][channel] = raws_img[i][channel].crop((x_left, y_top, x_right, y_bottom))


# In[20]:


test_data, file_paths = load_data(DATASET_PATH)
raws_img = load_as_images(get_raws(test_data))
original_size = raws_img[0][0].size[0]

# transform data according to the format it was trained on
crop(raws_img, box_size=32)  # trained on 64 x 32 x 32
pad = (original_size // 2) - (32 // 2)
# get data batch in tensor format
raws_tensor = torch.from_numpy(img_to_numpy(raws_img)).unsqueeze(dim=1).to(device)


# In[ ]:


# Load model
model = UnetModel(in_channels=1, out_channels=1, model_depth=3)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# Generate predictions
probas_tensor = model(raws_tensor)
preds = (probas_tensor > 0.5).long()
preds = preds.detach().cpu().numpy().squeeze()

for i, path in enumerate(file_paths):
    _path = Path(path)
    scan_name = _path.stem
    pred_name = 'pred_' + scan_name
    save_path = Path(PREDICTIONS_PATH).joinpath(pred_name)
    # pad array to undo crop
    padded_pred = np.zeros_like(preds[i], shape=(preds[i].shape[0], original_size, original_size), dtype=np.float32)
    for ch in range(padded_pred.shape[0]):
        padded_pred[ch][pad:-pad, pad:-pad] = preds[i][ch]
    np.save(str(save_path), padded_pred)
