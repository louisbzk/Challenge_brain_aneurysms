import numpy as np
import h5py
import os
from typing import List


def load_data(path: str) -> List[h5py.File]:
    data = []
    for root, dirs, filenames in os.walk(path):
        for file in filenames:
            data.append(h5py.File(f'{root}{file}'))

    return data


def get_raw_at(data: List[h5py.File], index: int) -> np.ndarray:
    return data[index]['raw']


def get_label_at(data: List[h5py.File], index: int) -> np.ndarray:
    return data[index]['label']


def get_raws(data: List[h5py.File]) -> np.ndarray:
    raws = np.zeros(shape=(len(data), *data[0]['raw'].shape), dtype=np.uint8)
    for i in range(len(raws)):
        raws[i] = data[i]['raw']

    return raws


def get_labels(data: List[h5py.File]) -> np.ndarray:
    labels = np.zeros(
        shape=(len(data), *data[0]['label'].shape), dtype=np.uint8)
    for i in range(len(labels)):
        labels[i] = data[i]['label']

    return labels
