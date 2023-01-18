from dataset import AneurysmDataset
from transforms import raw_transform, label_transform, joint_transform
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


TRANSFORM_PARAMS = {
    'raw': {
        'contrast_factor': 3.,
        'cluster_colors': 20,
        'cluster_kmeans': 0,
        'cluster_method': 0,
    },

    'label': {
        'clean_dist_thresh': 10.,
    },

    'joint': {
        'max_abs_rot': 180.,
        'sharpen_factor': 5.,
        'zoom_box': 65,
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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    raw, label = dataset[0]
    raw_arr, label_arr = raw.detach().cpu().numpy(), label.detach().cpu().numpy()
    plt.imshow(np.hstack((raw_arr[30], label_arr[30])), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main(batch_size=1, **TRANSFORM_PARAMS)
