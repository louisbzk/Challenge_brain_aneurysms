import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
from transforms import load_as_images


class AneurysmDataset(Dataset):
    """
    Torch dataset for our aneurysms.
    It allows loading data as tensors without loading all the data at once, but using minibatches. It also allows to
    apply transformations (ie data augmentations and distance transform for BoundaryLoss) as we iterate over the data
    """

    def __init__(self, dataset_path: str, joint_transform=None, raw_transform=None, label_transform=None):
        """
        Create the Dataset. The data is first loaded onto RAM to make sure we don't overrun the GPU,
        which is slower at training time

        :param dataset_path: path to dataset directory
        :param joint_transform: callable, a data augmentation transform performed on both raws and labels
        :param raw_transform: callable, a data augmentation transform performed on raws only
        :param label_transform: callable, a data augmentation transform performed on labels only
        """
        self.dataset_path = dataset_path
        self.joint_transform = joint_transform
        self.raw_transform = raw_transform
        self.label_transform = label_transform
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        try:
            _, _, files = next(os.walk(dataset_path))
        except StopIteration:
            raise OSError(f'Directory {dataset_path} is empty !')

        self.num_scans = len(files)

        # todo : precompute distance transform for all scans
        self.dist_transforms = None

    def __len__(self):
        return self.num_scans

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(f'Sample of index {idx} was asked, but there are {len(self)} samples !')

        file_path = os.path.join(self.dataset_path, f'scan_{idx+1}.h5')
        h5f = h5py.File(file_path)
        raw, label = load_as_images(np.array(h5f['raw']), np.array(h5f['label']))

        transform_hist = None
        if self.raw_transform:
            raw = self.raw_transform(raw)
        if self.label_transform:
            label = self.label_transform(label)
        if self.joint_transform:
            raw, label, transform_hist = self.joint_transform(raw, label)

        raw = torch.from_numpy(raw).unsqueeze(0)  # add 1 dimension
        label = torch.from_numpy(label).unsqueeze(0)

        return raw, label  # , file_path, transform_hist
