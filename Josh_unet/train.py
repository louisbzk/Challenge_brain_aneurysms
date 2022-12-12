# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 27/08/2019 08:54
import torch
from torch.nn import CrossEntropyLoss
from unet.unet3d import UnetModel, Trainer
# from unet3d_model.tmp import UNet
from unet.loss import DiceLoss
import numpy as np
import h5py
import os
# from dataviz import view_sample, show_aneurysm
import matplotlib.pyplot as plt

def batch_data_gen(pet_imgs, mask_imgs, iter_step, batch_size=6):
    """
    Get training batch to feed convolution neural network
    :param pet_imgs: the whole batch of pet images
    :param mask_imgs: the whole batch of mask images
    :param iter_step: the iteration step during training process
    :param batch_size: batch size to generate
    :return: batch images and batch masks
    """
    # shuffling data
    permutation_idxs = np.random.permutation(len(pet_imgs))
    pet_imgs = pet_imgs[permutation_idxs]
    mask_imgs = mask_imgs[permutation_idxs]

    # count iteration step to get corresponding training batch
    step_count = batch_size * iter_step
    print(step_count)
    return pet_imgs[step_count: batch_size + step_count], mask_imgs[step_count: batch_size + step_count]


def train_main(x,y, in_channels, out_channels, learning_rate, no_epochs):
    """
    Train module
    :param data_folder: data folder
    :param in_channels: the input channel of input images
    :param out_channels: the final output channel
    :param learning_rate: set learning rate for training
    :param no_epochs: number of epochs to train model
    :return: None
    """
    model = UnetModel(in_channels=in_channels, out_channels=out_channels)
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = DiceLoss()
    trainer = Trainer(net=model, optimizer=optim, criterion=criterion, no_epochs=no_epochs)
    trainer.train(batch_data_loader=batch_data_gen, x=x, y=y)


if __name__ == "__main__":
    data = []
    for root, dirs, filenames in os.walk('challenge_dataset/'):  # adapt path
        for file in filenames:
            data.append(h5py.File(f'{root}{file}'))

    # print(f'Data shape : \n'
    #   f'Raws : {data[0]["raw"].shape}\n'
    #   f'Labels : {data[0]["label"].shape}\n'
    #   f'Data type : \n'
    #   f'Raws : {data[0]["raw"].dtype}\n'
    #   f'Labels : {data[0]["label"].dtype}')

    x = []
    y = []
    for i in range(len(data)):
        x.append(np.array(data[i]["raw"], dtype=np.float32))
        y.append(np.array(data[i]["label"], dtype=np.float32))
    x = np.array(x)
    y = np.array(y)
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    print(x.shape, y.shape)
    train_main(x[:8],y[:8], in_channels=1, out_channels=1, learning_rate=0.0001, no_epochs=10)
