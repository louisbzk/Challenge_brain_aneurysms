import torch.optim

from dataset import AneurysmDataset
from transforms import raw_transform, label_transform, joint_transform, undo_transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from unet.unet3d import UnetModel, Trainer
from unet.loss import DiceLoss, DiceBCEFocalLoss
from pathlib import Path


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


def simple_eval_loop(n_eval, model, dataloader, save_path):
    for i in range(n_eval):
        sample = next(iter(dataloader))
        raw, label = sample[0].to(model.device), sample[1].to(model.device)
        sample_file = Path(sample[2])
        sample_transforms = sample[3]

        pred = model(raw).detach().cpu().numpy()
        pred = undo_transforms(pred, sample_transforms)
        label = label.detach().cpu().numpy()
        full_save_path = Path(save_path).joinpath(sample_file.name)
        np.save(str(full_save_path), pred)
        np.save(str(full_save_path), label)
    return


def main(batch_size, n_epochs, n_eval, load_path, eval_save_path, **kwargs):
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
    if load_path:
        model.load_state_dict(torch.load(load_path))
        if n_eval > 0:
            model.eval()
            simple_eval_loop(n_eval, model, dataloader, eval_save_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = DiceBCEFocalLoss(dice_bce_weight=0.6)

    trainer = Trainer(net=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      no_epochs=n_epochs,
                      batch_size=batch_size)

    loss_history = trainer.train(dataloader)
    plt.plot(loss_history)
    plt.show()

    torch.save(model.state_dict(), f'models/DiceBCEFocal/DiceBCEFocal_w0.6_gamma2_ep{n_epochs}.pth')
    model.eval()

    for i in range(n_eval):
        sample = next(enumerate(dataloader))
        raw, label = sample[0].to(device), sample[1].to(device)

        pred = model(raw).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        np.save(f'models/DiceBCEFocal/DiceBCEFocal_w0.6_gamma2_ep{n_epochs}_pred{i}.npy', pred)
        np.save(f'models/DiceBCEFocal/DiceBCEFocal_w0.6_gamma2_ep{n_epochs}_true{i}.npy', label)


if __name__ == '__main__':
    main(batch_size=1, n_epochs=10, n_eval=10,
         load_path='models/DiceBCEFocal/DiceBCEFocal_w0.6_gamma2_ep10.pth',
         eval_save_path='models/DiceBCEFocal/DiceBCEFocal_w0.6_gamma2_ep10', **TRANSFORM_PARAMS)
