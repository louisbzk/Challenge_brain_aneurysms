import torch.optim

from dataset import AneurysmDataset
from transforms import raw_transform, label_transform, joint_transform
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from unet.unet3d import UnetModel, Trainer
from unet.loss import DiceLoss


# set variables to None to skip the corresponding transform
TRANSFORM_PARAMS = {
    'raw': {
        'contrast_factor': 3.5,
        'cluster_colors': None,
        'cluster_kmeans': 30,
        'cluster_method': 0,
    },

    'label': {
        'clean_dist_thresh': 10.,
    },

    'joint': {
        'max_abs_rot': 45.,
        'flip': True,
        'sharpen_factor': 1.2,
        'zoom_box': 101,
    }
}


def main(batch_size, **kwargs):
    def joint_transform_fixed(raws, labels): return joint_transform(raws, labels, **TRANSFORM_PARAMS['joint'])
    def raw_transform_fixed(raws): return raw_transform(raws, **TRANSFORM_PARAMS['raw'])
    def label_transform_fixed(labels): return label_transform(labels, **TRANSFORM_PARAMS['label'])

    dataset = AneurysmDataset('challenge_dataset',
                              joint_transform=joint_transform_fixed,
                              raw_transform=raw_transform_fixed,
                              label_transform=label_transform_fixed)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = UnetModel(in_channels=1, out_channels=1, model_depth=3).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = DiceLoss(epsilon=1e-8)

    trainer = Trainer(net=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      no_epochs=50,
                      batch_size=batch_size)

    trainer.train(dataloader)


if __name__ == '__main__':
    main(batch_size=1, **TRANSFORM_PARAMS)
