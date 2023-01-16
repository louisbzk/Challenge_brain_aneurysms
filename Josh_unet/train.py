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
from dataviz import view_sample, show_aneurysm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def save_data(x, y, PATH):
    torch.save({'x': x, 'y': y}, PATH)


def load_data(PATH):
    data = torch.load(PATH)
    return data['x'], data['y']


def save_model(model, optimizer, epoch, loss, lr, PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'learning_rate': lr
    }, PATH)


def load_model(PATH, eval=True):
    checkpoint = torch.load(PATH)
    model = UnetModel(in_channels=1, out_channels=1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.to(torch.device('cuda'))

    if eval:
        model.eval()
    # - or -
    else:
        model.train()
    return model, optimizer, loss, epoch


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
    return pet_imgs[step_count: batch_size + step_count], mask_imgs[step_count: batch_size + step_count]


def train_main(x, y, in_channels, out_channels, learning_rate, no_epochs):
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
    loss_array = trainer.train(batch_data_loader=batch_data_gen, x=x, y=y)
    save_model(model, optim, no_epochs, loss_array, learning_rate, 'holaBuenoDias')
    return model, optim, loss_array, epoch


if __name__ == '__main__':
    # data = []
    # for root, dirs, filenames in os.walk('C:\\Users\\33638\\Documents\\Mines-Paris\\Challenge_brain_aneurysms\\Josh_unet\\challenge_dataset\\'):  # adapt path
    #     for file in filenames:
    #         data.append(h5py.File(f'{root}{file}'))

    # # print(f'Data shape : \n'
    # #   f'Raws : {data[0]["raw"].shape}\n'
    # #   f'Labels : {data[0]["label"].shape}\n'
    # #   f'Data type : \n'
    # #   f'Raws : {data[0]["raw"].dtype}\n'
    # #   f'Labels : {data[0]["label"].dtype}')

    # x = []
    # y = []
    # for i in range(len(data)):
    #     x.append(np.array(data[i]["raw"], dtype=np.float32))
    #     y.append(np.array(data[i]["label"], dtype=np.float32))
    # x = np.array(x)
    # y = np.array(y)
    # x = np.expand_dims(x, axis=1)
    # y = np.expand_dims(y, axis=1)

    train = False
    x, y = load_data('FormattedData')

    if train:
        model, optimizer, loss, epoch = train_main(
            x[:], y[:], in_channels=1, out_channels=1, learning_rate=0.01, no_epochs=10)
    else:
        model, optimizer, loss, epoch = load_model('holaBuenoDias', eval=True)

    newY = model(torch.from_numpy(x[:1]).cuda()).cpu().detach().numpy()

    z_plot, x_plot, y_plot = (newY[0, 0] > 0.7).nonzero()
    vein_data = go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        marker=go.scatter3d.Marker(size=1),
        opacity=1.0,
        mode='markers')

    z_plot, x_plot, y_plot = (y[0, 0] > 0.7).nonzero()
    marker_data = go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        marker=go.scatter3d.Marker(size=3),
        opacity=1.0,
        mode='markers')

    fig = go.Figure(data=[marker_data, vein_data])
    fig.show()
