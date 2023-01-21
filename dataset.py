from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Optional


class AneurysmDataset(Dataset):
    def __init__(self, dataset_path: str, crop: Optional[int] = None):
        self.dataset_path = dataset_path

        self.raws = []
        self.labels = []
        self.fpaths = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                h5data = h5py.File(os.path.join(root, file))
                self.raws.append(np.array(h5data['raw'], dtype=np.float32))
                self.labels.append(np.array(h5data['label'], dtype=np.float32))
                self.fpaths.append(Path(file))
                if np.max(self.labels[-1]) == 0.:
                    raise ValueError(f'at scan file \'{self.fpaths[-1].name}\', label is only zeros')

        self.crop = crop
        img_size = self.raws[0][0].shape[0]
        img_center = (img_size // 2, img_size // 2)

        if self.crop:
            # top left
            self.crop_tl = (img_center[0] - self.crop//2, img_center[1] - self.crop//2)
            # bottom right
            self.crop_br = (img_center[0] + self.crop//2, img_center[1] + self.crop//2)
        else:
            self.crop_tl = None
            self.crop_br = None

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(f'Sample of index {idx} was asked, but there are {len(self)} samples !')

        raw = self.raws[idx]
        label = self.labels[idx]

        if self.crop:
            raw = raw[:, self.crop_tl[0]:self.crop_br[0], self.crop_tl[1]:self.crop_br[1]]
            label = label[:, self.crop_tl[0]:self.crop_br[0], self.crop_tl[1]:self.crop_br[1]]

        raw = torch.from_numpy(raw).unsqueeze(0)  # add 1 dimension
        label = torch.from_numpy(label).unsqueeze(0)

        return raw, label


def dataset_test():
    dataset = AneurysmDataset('augmented_data', crop=8)
    dataset[0]

    dataset = AneurysmDataset('augmented_data')
    dataset[0]


if __name__ == '__main__':
    dataset_test()
