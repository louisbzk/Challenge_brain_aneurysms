import torch.optim

from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import numpy as np
from unet.unet3d import UnetModel
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.nn import functional as F
import h5py
import os
from tqdm import tqdm
import time
from unet.loss import IoUCohensKappa, DiceLoss


PROCESSED_DATA_PATH = 'augmented_data/'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProcessedDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        self.raws = []
        self.labels = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                h5data = h5py.File(f'{root}{file}')
                self.raws.append(np.array(h5data['raw'], dtype=np.float32))
                self.labels.append(np.array(h5data['label'], dtype=np.float32))

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(f'Sample of index {idx} was asked, but there are {len(self)} samples !')

        raw = self.raws[idx]
        label = self.labels[idx]

        raw = torch.from_numpy(raw).unsqueeze(0)  # add 1 dimension
        label = torch.from_numpy(label).unsqueeze(0)

        return raw, label


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = weight

    def forward(self, inputs, targets, smooth=1e-8):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (1. - self.dice_weight) * BCE + self.dice_weight * dice_loss

        return Dice_BCE


def mean_iou(_logits, labels, smooth=1e-8):
    inputs = _logits.view(-1)
    targets = labels.view(-1)

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return IoU


p_dataset = ProcessedDataset(PROCESSED_DATA_PATH)
batch_size = 1
validation_split = .2
n_epochs = 200

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(f'Training on device : {device}')
_pin_mem = device.type == 'cuda'

dataset_size = len(p_dataset)
all_indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

train_indices, val_indices = all_indices[split:], all_indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_data_loader = DataLoader(p_dataset, batch_size, shuffle=False, sampler=train_sampler,
                               pin_memory=_pin_mem)

val_data_loader = DataLoader(p_dataset, batch_size, shuffle=False, sampler=val_sampler,
                             pin_memory=_pin_mem)

model = UnetModel(in_channels=1, out_channels=1, model_depth=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = IoUCohensKappa(iou_weight=0.85)

loss_array = []
iou_array = []
iter_per_epoch = len(train_data_loader)  # len(dataset) / batch_size
for epoch in range(n_epochs):
    print('Epoch no: ', epoch)
    start_time = time.time()
    train_losses, train_iou = 0, 0
    val_losses, val_iou = 0, 0
    loss_array.append([])
    iou_array.append([])

    model.train()
    for idx, batch in tqdm(enumerate(train_data_loader), total=iter_per_epoch):
        x_batch = batch[0].to(device)
        y_batch = batch[1].to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        train_iou += mean_iou(logits, y_batch).item()
        train_losses += loss.item()

    model.eval()
    for idx, batch in enumerate(val_data_loader):
        x_batch = batch[0].to(device)
        y_batch = batch[1].to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        val_losses += loss.item()
        val_iou += mean_iou(logits, y_batch).item()

    loss_array[-1].append(train_losses / iter_per_epoch)
    iou_array[-1].append(train_iou / iter_per_epoch)

    loss_array[-1].append(val_losses / len(val_data_loader))
    iou_array[-1].append(val_iou / len(val_data_loader))

    print('Rebuilding Data Loaders...')
    np.random.shuffle(all_indices)
    train_indices, val_indices = all_indices[split:], all_indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_data_loader = DataLoader(p_dataset, batch_size, shuffle=False, sampler=train_sampler,
                                   pin_memory=_pin_mem)

    val_data_loader = DataLoader(p_dataset, batch_size, shuffle=False, sampler=val_sampler,
                                 pin_memory=_pin_mem)

    if epoch % 20 == 0:
        print('Saving checkpoint...')
        torch.save(model.state_dict(),
                   f'models/IoUCohen/IoUCohen_checkpoint.pth')

    end_time = time.time()

    print('Epoch {}, training loss {:.4e}, val loss {:.4e}, training IoU {:.4e}, val IoU {:.4e}, time {:.2f}'.format(
        epoch, train_losses / iter_per_epoch, val_losses / len(val_data_loader),
        train_iou / iter_per_epoch, val_iou / len(val_data_loader), end_time - start_time
    )
    )


loss_array = np.array(loss_array)
iou_array = np.array(iou_array)
torch.save(model.state_dict(), f'models/IoUCohen/IoUCohen_checkpoint.pth')

_, ax = plt.subplots(2, 2, figsize=(12, 12))

ax[0][0].plot(loss_array[:, 0], label='train_loss')
ax[0][0].legend()
ax[0][1].plot(loss_array[:, 1], label='val_loss')
ax[0][1].legend()

ax[1][0].plot(iou_array[:, 0], label='train_iou')
ax[1][0].legend()
ax[1][1].plot(iou_array[:, 1], label='val_iou')
ax[1][1].legend()

plt.show()
