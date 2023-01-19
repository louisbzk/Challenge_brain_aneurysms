# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:52
from tqdm import tqdm
from unet.building_components import EncoderBlock, DecoderBlock
import sys
import time
import torch
import torch.nn as nn
sys.path.append('..')


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


class Trainer(object):

    def __init__(self, net, optimizer, criterion, no_epochs, batch_size=1):
        """
        Parameter initialization
        :param data_dir: folder that stores images for each modality
        :param net: the created model
        :param optimizer: the optimizer mode
        :param criterion: loss function
        :param no_epochs: number of epochs to train the model
        :param batch_size: batch size for generating data during training
        """
        self.modalities = ['PET', 'MASK']
        self.net = net
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            self.net.cuda()
        self.optimizer = optimizer
        self.criterion = criterion
        self.no_epochs = no_epochs
        self.batch_size = batch_size

    def train(self, train_data_loader):
        """
        Load corresponding data and start training

        :param train_data_loader: DataLoader object
        :return: None
        """
        loss_array = []
        iter_per_epoch = len(train_data_loader)  # len(dataset) / batch_size
        for epoch in range(self.no_epochs):
            print('Epoch no: ', epoch)
            start_time = time.time()
            train_losses, train_iou = 0, 0
            for idx, batch in tqdm(enumerate(train_data_loader), total=iter_per_epoch):
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                self.optimizer.zero_grad()

                logits = self.net(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                # train_iou += mean_iou(y_batch, logits)
                train_losses += loss.item()
            end_time = time.time()
            loss_array.append(train_losses / iter_per_epoch)
            print('Epoch {}, training loss {:.4f}, time {:.2f}'.format(epoch, train_losses / iter_per_epoch,
                                                                       end_time - start_time))
        return loss_array

    def predict(self):
        pass

    def _save_checkpoint(self):
        pass


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    inputs = torch.randn(1, 1, 96, 96, 96)
    print('The shape of inputs: ', inputs.shape)
    data_folder = '../processed'
    model = UnetModel(in_channels=1, out_channels=1)
    if device.type == 'cuda':
        inputs = inputs.cuda()
        model.cuda()
    x = model(inputs)
    print(model)
