import torch
import torch.nn
import os
import h5py
import numpy as np
from DAResUNet.daresunet import DAResUNet
from torchviz import make_dot
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--force-cpu', '-c', action='store_true',
                    help='Force CPU device ?')


def main():
    args = parser.parse_args()
    if args.force_cpu:
        device = torch.device('cpu')
        print(f'\033[93m'
              f'Forcing CPU usage\n'
              f'Running on CPU, model may be slow'
              f'\033[0m')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'\033[92m'
              f'CUDA available, running on device \'{torch.cuda.get_device_name(device)}\''
              f'\033[0m')
    else:
        device = torch.device('cpu')
        print(f'\033[93m'
              f'CUDA not available !\n'
              f'Running on CPU, model may be slow'
              f'\033[0m')

    data = []
    for root, dirs, filenames in os.walk('challenge_dataset/'):  # adapt path
        for file in filenames:
            data.append(h5py.File(f'{root}{file}'))

    data_shape = data[0]['raw'].shape
    raws = np.expand_dims(np.array([data[i]['raw']
                          for i in range(len(data))]), axis=1)
    raws = torch.as_tensor(raws, dtype=torch.uint8, device=device)

    labels = np.expand_dims(
        np.array([data[i]['label'] for i in range(len(data))]), axis=1)
    labels = torch.as_tensor(labels, dtype=torch.uint8, device=device)

    print(
        f'\033[92m'
        f'Succesfully loaded tensors of size {raws.size()} to device\n'
        f'Total memory : {"{:.2e}".format(np.prod(raws.size())) * 1} bytes'
        f'\033[0m'
    )

    model = DAResUNet().to(device)
    dummy = torch.zeros(size=[1, 1, 64, 192, 192],
                        dtype=torch.float32, device=device)
    y_dummy = model(dummy)['y']
    make_dot(y_dummy, params=dict(list(model.named_parameters()))
             ).render('Model plot', format='png')


if __name__ == '__main__':
    main()
