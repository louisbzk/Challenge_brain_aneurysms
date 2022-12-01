import numpy as np
import h5py
import os
from dataviz import view_sample, show_aneurysm
import matplotlib.pyplot as plt


data = []
for root, dirs, filenames in os.walk('challenge_dataset/'):  # adapt path
    for file in filenames:
        data.append(h5py.File(f'{root}{file}'))

print(f'Data shape : \n'
      f'Raws : {data[0]["raw"].shape}\n'
      f'Labels : {data[0]["label"].shape}\n'
      f'Data type : \n'
      f'Raws : {data[0]["raw"].dtype}\n'
      f'Labels : {data[0]["label"].dtype}')

# Is there 1 and only 1 aneurysm per sample ?
no_aneurysm_samples = 0
multiple_aneurysms = 0

for sample_idx, sample in enumerate(data):
    max_arr = np.zeros(shape=len(sample['label']))
    for lab_idx, label in enumerate(sample['label']):
        max_arr[lab_idx] = label.max()
    labeled = np.trim_zeros(max_arr)
    if labeled.size == 0:
        print(f'At sample {sample_idx}, there is no aneurysm')
        no_aneurysm_samples += 1
    if labeled.min() == 0:
        print(f'At sample {sample_idx}, there is more than one aneurysm')
        multiple_aneurysms += 1

print(f'\nTotal irregular samples : {no_aneurysm_samples + multiple_aneurysms}\n'
      f'Samples with no aneurysm : {no_aneurysm_samples}\n'
      f'Samples with more than one aneurysm : {multiple_aneurysms}')

view_sample(data, 5, 'label')
plt.show()

show_aneurysm(data, 0, plot_size=3)
plt.show()

