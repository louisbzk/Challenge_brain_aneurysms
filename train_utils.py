from dataset import AneurysmDataset
from torch.utils.data import DataLoader
import torch
import os
import pickle


def get_loaders(train_path, val_path, num_workers, batch_size, pin_memory):
    train_dataset = AneurysmDataset(train_path, crop=None)
    val_dataset = AneurysmDataset(val_path, crop=None)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=pin_memory)

    return train_loader, val_loader


def save_checkpoint(model, optimizer, path):
    torch.save(model.state_dict(), os.path.join(path, 'model_checkpoint.pth'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_checkpoint.pth'))


def load_checkpoint(model, optimizer, path):
    model_state_dict = torch.load(os.path.join(path, 'model_checkpoint.pth'))
    optim_state_dict = torch.load(os.path.join(path, 'optimizer_checkpoint.pth'))

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optim_state_dict)


def save_histories(hist_dict, path):
    with open(path, 'wb') as f:
        pickle.dump(hist_dict, f)


def load_histories(path):
    with open(path, 'rb') as f:
        hist_dict = pickle.load(f)

    return hist_dict
