from multipledispatch import dispatch
import matplotlib.pyplot as plt
import numpy as np


@dispatch(list, int, str)
def view_sample(imgs, idx, kind='raw'):
    """
    View an entire sample given an index.
    :param imgs: data
    :param idx: index of sample
    :param kind: 'raw' for brain image, 'label' for aneurysm label
    :return: None
    """
    _, _ax = plt.subplots(8, 8, figsize=(64, 64))

    for i in range(8):
        for j in range(8):
            _ax[i][j].imshow(imgs[idx][kind][8 * i + j], cmap='gray')


@dispatch(np.ndarray, int)
def view_sample(array, idx):
    _, _ax = plt.subplots(8, 8, figsize=(64, 64))

    for i in range(8):
        for j in range(8):
            _ax[i][j].imshow(array[idx][8 * i + j], cmap='gray')


@dispatch(list, int, plot_size=int, img_cmap=str, label_cmap=str)
def show_aneurysm(imgs,
                  idx: int,
                  plot_size=6,
                  img_cmap='gray',
                  label_cmap='gray'):
    """
    Given an index, plot images of the aneurysm
    :param imgs: List of hdf5 data
    :param idx: int in [0, 105]
    :param plot_size: int, size of each square subplot
    :param label_cmap: str or matplotlib color map object
    :param img_cmap: str or matplotlib color map object
    :return: None
    """
    print("here 1")  # jingwei
    raw, label = imgs[idx]['raw'], imgs[idx]['label']
    _show_aneurysm_raw_label(raw, label, img_cmap, label_cmap, plot_size)


@dispatch(np.ndarray, np.ndarray, int, plot_size=int, img_cmap=str, label_cmap=str)
def show_aneurysm(raws: np.ndarray,
                  labels: np.ndarray,
                  idx: int,
                  plot_size=6,
                  img_cmap='gray',
                  label_cmap='gray'):
    """
    Given an index, plot images of the aneurysm
    :param raws: array of raws
    :param labels: array of labels
    :param idx: int in [0, 105]
    :param plot_size: int, size of each square subplot
    :param label_cmap: str or matplotlib color map object
    :param img_cmap: str or matplotlib color map object
    :return: None
    """
    print("here 2")  # jingwei
    raw, label = raws[idx], labels[idx]
    _show_aneurysm_raw_label(raw, label, img_cmap, label_cmap, plot_size)


@dispatch(np.ndarray, np.ndarray, int)
def show_aneurysm(raws: np.ndarray,
                  labels: np.ndarray,
                  idx: int):
    """
    Given an index, plot images of the aneurysm
    :param raws: array of raws
    :param labels: array of labels
    :param idx: int in [0, 105]
    :return: None
    """
    print("here 3")  # jingwei
    show_aneurysm(raws, labels, idx, 6, 'gray', 'gray')


def _show_aneurysm_raw_label(raw: np.ndarray,
                             label: np.ndarray,
                             img_cmap: str,
                             label_cmap: str,
                             plot_size: int):
    i_min, i_max = 0, -1

    # Find first nonzero label
    for i, lab in enumerate(label):
        if lab.max() != 0:
            i_min = i
            break

    # Find last nonzero label
    for i, lab in enumerate(reversed(label)):
        if lab.max() != 0:
            i_max = -i
            break

    # Take 1 image before and after the labeled aneurysm
    if i_min != 0:
        i_min -= 1
    if i_max != -1:
        i_max += 1
    i_max = len(label) + i_max

    n_imgs = i_max - i_min + 1
    _, axes = plt.subplots(2, n_imgs, figsize=(
        plot_size, n_imgs * plot_size // 2))
    for i in range(n_imgs):
        axes[i][0].imshow(raw[(i + i_min)%len(label)], cmap=img_cmap)
        axes[i][1].imshow(label[(i + i_min)%len(label)], cmap=label_cmap)

    plt.show()
