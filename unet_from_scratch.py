import torch
import torch.nn as nn
from torchvision.transforms import functional as TF


class DoubleConv(nn.Module):
    """
    The basic conv block of UNet. 2 convs at each block.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # 1st conv
            # 'same' convolution (keep height, width) + no bias because we use BatchNorm (cancels bias)
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # 2nd conv
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features_sizes=(64, 128, 256, 512)):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down (encoding) part
        # double conv -> double conv -> ...
        for feat_size in features_sizes:
            self.downs.append(DoubleConv(in_channels, feat_size))
            in_channels = feat_size

        # Up (decoding) part
        # upsampling conv -> double conv -> upsampling conv ...
        for feat_size in reversed(features_sizes):
            self.ups.append(
                # in : 2 * feat_size because we will concatenate the outputs of the downs
                nn.ConvTranspose3d(in_channels=2*feat_size, out_channels=feat_size, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(in_channels=2*feat_size, out_channels=feat_size))

        # bottom part (feature concatenation)
        self.bottom = DoubleConv(features_sizes[-1], 2*features_sizes[-1])

        self.final_conv = nn.Conv3d(features_sizes[0], out_channels, kernel_size=1)

    def forward(self, x):
        feat_to_concat = []  # stored from highest level to lowest level
        for down in self.downs:
            x = down(x)
            feat_to_concat.append(x)
            x = self.pool(x)

        x = self.bottom(x)
        feat_to_concat = feat_to_concat[::-1]

        for i in range(0, len(self.ups), 2):  # step size 2 : we have ConvTranspose and DoubleConv in the up modules
            x = self.ups[i](x)  # do the ConvTranspose
            prev_feature = feat_to_concat[i // 2]
            # the dimension of x and feat_concat may not always match, if the input size is not divisible by 16
            # uncomment if shape issues
            # if x.shape != prev_feature.shape:
            #     x = TF.resize(x, size=prev_feature.shape[2:])
            feat_concat = torch.cat((prev_feature, x), dim=1)  # concat along the channel dimension
            x = self.ups[i+1](feat_concat)  # do the DoubleConv

        return self.final_conv(x)


def unet_test():
    x = torch.randn((3, 1, 64, 32, 32))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == '__main__':
    unet_test()
