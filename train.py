from tqdm import tqdm
from unet_from_scratch import UNet
import torch
from loss import DiceFocalLogits, IoUCohensKappa, IoUFocalLogits
from metrics import mean_iou_logits
from time import time
import os
from train_utils import get_loaders, save_checkpoint, load_checkpoint, save_histories, load_histories
from matplotlib import pyplot as plt


LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'DEVICE : {DEVICE.type}')
BATCH_SIZE = 8
NUM_EPOCHS = 200
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == 'cuda'
TRAIN_SET_PATH = 'train_augmented_data'
VAL_SET_PATH = 'val_augmented_data'
CRITERION = IoUCohensKappa(iou_weight=0.8)
CHECKPOINT_ITER = 25
CHECKPOINT_PATH = 'checkpoint/IoUCohen_08'
SAVE_PATH = 'models/IoUCohen_08'


def main():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    criterion = CRITERION
    train_loader, val_loader = get_loaders(
        TRAIN_SET_PATH, VAL_SET_PATH, NUM_WORKERS, BATCH_SIZE, PIN_MEMORY
    )

    train_batches = len(train_loader)
    val_batches = len(val_loader)

    train_loss_hist = []
    val_loss_hist = []
    train_iou_list = []
    val_iou_list = []
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch : {epoch}/{NUM_EPOCHS}')
        start = time()
        loop = tqdm(train_loader)

        mean_train_loss = 0
        mean_val_loss = 0
        mean_train_iou = 0
        mean_val_iou = 0

        model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(loop):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # forward
            with torch.autocast(device_type=DEVICE.type):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # metrics
            mean_train_loss += loss.item()
            mean_train_iou += mean_iou_logits(logits, y_batch)

        train_loss_hist.append(mean_train_loss / train_batches)
        train_iou_list.append(mean_train_iou / train_batches)

        # validation
        model.eval()
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            with torch.no_grad():
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            mean_val_loss += loss.item()
            mean_val_iou += mean_iou_logits(logits, y_batch)

        val_loss_hist.append(mean_val_loss / val_batches)
        val_iou_list.append(mean_val_iou / val_batches)

        end = time()

        if epoch % CHECKPOINT_ITER == 0:
            print('Saving checkpoint...')
            save_checkpoint(model, optimizer, CHECKPOINT_PATH)
            save_histories({
                'train_loss': train_loss_hist,
                'val_loss': val_loss_hist,
                'train_iou': train_iou_list,
                'val_iou': val_iou_list,
            }, os.path.join(SAVE_PATH, 'history.pickle'))

        print(f'train_loss : {"{:.2e}".format(train_loss_hist[-1])}, '
              f'val_loss : {"{:.2e}".format(val_loss_hist[-1])}, '
              f'train IoU : {"{:.2e}".format(train_iou_list[-1])}, '
              f'val IoU : {"{:.2e}".format(val_iou_list[-1])}, '
              f'time : {"{:.2f}".format(end - start)}s')

    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(SAVE_PATH, 'optim.pth'))
    save_histories({
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'train_iou': train_iou_list,
        'val_iou': val_iou_list,
    }, os.path.join(SAVE_PATH, 'history.pickle'))

    return train_loss_hist, val_loss_hist, train_iou_list, val_iou_list


def test_save():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss, val_loss, train_iou, val_iou = [1], [2], [3], [4]

    save_checkpoint(model, optimizer, CHECKPOINT_PATH)

    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(SAVE_PATH, 'optim.pth'))
    save_histories({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_iou': train_iou,
        'val_iou': val_iou,
    }, os.path.join(SAVE_PATH, 'history.pickle'))


if __name__ == '__main__':
    train_loss, val_loss, train_iou, val_iou = main()
    _, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].plot(train_loss, label='train_loss')
    ax[0].plot(val_loss, label='val_loss')
    ax[0].legend()

    ax[1].plot(train_iou, label='train_iou')
    ax[1].plot(val_iou, label='val_iou')
    ax[1].legend()

    plt.show()
    plt.savefig(os.path.join(SAVE_PATH, 'cv_plot'), format='png')
