"""
ORIGINAL CODE FROM : https://github.com/deepwise-code/DLIA/blob/main/tasks/aneurysm/nets/resunet.py
PAPER : https://www.nature.com/articles/s41467-020-19527-w
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from .attention import PAMModule, CAMModule


def norm(planes, mode='bn', groups=12):
    if mode == 'bn':
        return nn.BatchNorm3d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()


class CBR(nn.Module):
    """
    Conv + Batch Normalization + ReLU
    """

    def __init__(self, n_in, n_out, kernel_size=(3, 3, 3), stride=1, dilation=1):
        super(CBR, self).__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        padding = (
            int((kernel_size[0] - 1) / 2) * dilation, int((kernel_size[1] - 1) / 2) * dilation, int((kernel_size[2] - 1) / 2) * dilation)
        self.conv = nn.Conv3d(n_in, n_out, kernel_size, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
        self.bn = norm(n_out)
        self.act = nn.ReLU(True)

    def forward(self, _input):
        output = self.conv(_input)
        output = self.bn(output)
        output = self.act(output)

        return output


class CB(nn.Module):
    """
    Conv + Batch Normalization
    """

    def __init__(self, n_in, n_out, kernel_size=(3, 3, 3), stride=1, dilation=1):
        super(CB, self).__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        padding = (
            int((kernel_size[0] - 1) / 2) * dilation,
            int((kernel_size[1] - 1) / 2) * dilation,
            int((kernel_size[2] - 1) / 2) * dilation)
        self.conv = nn.Conv3d(n_in, n_out, kernel_size, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
        if n_out == 1:
            self.bn = nn.BatchNorm3d(n_out, momentum=0.95, eps=1e-03)
        else:
            self.bn = norm(n_out)

    def forward(self, _input):
        output = self.conv(_input)
        output = self.bn(output)

        return output


class C(nn.Module):
    """
    Conv (with appropriate padding computation)
    """

    def __init__(self, n_in, n_out, kernel_size=(3, 3, 3), stride=1, dilation=1):
        super(C, self).__init__()
        padding = (
            int((kernel_size[0] - 1) / 2) * dilation,
            int((kernel_size[1] - 1) / 2) * dilation,
            int((kernel_size[2] - 1) / 2) * dilation)
        self.conv = nn.Conv3d(n_in, n_out, kernel_size, stride=stride, padding=padding,
                              bias=False, dilation=dilation)

    def forward(self, _input):
        return self.conv(_input)


class BR(nn.Module):
    """
    Batch Normalization + ReLU
    """

    def __init__(self, n_in):
        super(BR, self).__init__()

        self.bn = norm(n_in)
        self.act = nn.ReLU(True)

    def forward(self, _input):
        return self.act(self.bn(_input))


class BasicBlock(nn.Module):
    """
    Perform the following operations :
    input -> Conv3D + BatchNorm + ReLU -> Conv3D + BatchNorm -> output

    output = ReLU(input + output)
    """
    expansion = 1

    def __init__(self, n_in, n_out, kernel_size=(3, 3, 3), prob=0.03, stride=1, dilation=1):

        super(BasicBlock, self).__init__()

        self.c1 = CBR(n_in, n_out, kernel_size, stride, dilation)
        self.c2 = CB(n_out, n_out, kernel_size, 1, dilation)
        self.act = nn.ReLU(True)

        self.downsample = None
        if n_in != n_out or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1,
                          stride=stride, bias=False),
                norm(n_out)
            )

    def forward(self, _input):
        output = self.c1(_input)
        output = self.c2(output)
        if self.downsample is not None:
            _input = self.downsample(_input)

        output = output + _input
        output = self.act(output)

        return output


class DownSample(nn.Module):
    def __init__(self, n_in, n_out, pool='max'):
        super(DownSample, self).__init__()

        if pool == 'conv':
            self.pool = CBR(n_in, n_out, 3, 2)
        elif pool == 'max':
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.pool = pool
            if n_in != n_out:
                self.pool = nn.Sequential(pool, CBR(n_in, n_out, 1, 1))

    def forward(self, _input):
        output = self.pool(_input)
        return output


class Upsample(nn.Module):
    def __init__(self, n_in, n_out):
        super(Upsample, self).__init__()
        self.conv = CBR(n_in, n_out)

    def forward(self, x):
        p = F.upsample(x, scale_factor=2, mode='trilinear')
        return self.conv(p)


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4

        # Conv 3D -> Batch Norm 3D -> ReLU
        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        # Conv 3D -> Batch Norm 3D -> ReLU
        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        # Attention
        self.sa = PAMModule(inter_channels)
        self.sc = CAMModule(inter_channels)

        # Conv 3D -> Batch Norm 3D -> ReLU
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        # Conv 3D -> Batch Norm 3D -> ReLU
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        # Dropout (5%), Conv3D, ReLU
        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

    def forward(self, x):
        # CBR -> Position attention module -> CBR -> (Dropout (5%), Conv3D, ReLU)
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        # CBR -> Channel attention module -> CBR -> (Dropout (5%), Conv3D, ReLU)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)

        return sasc_output
