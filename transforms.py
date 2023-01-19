import torch
from torch import Tensor
import numpy as np
import scipy.ndimage as scim
from PIL import Image, ImageEnhance


def load_as_images(raw_3d: np.ndarray, label_3d: np.ndarray):
    """
    From a given raw of shape (channels, width, height), return arrays of Image of shape (channels)

    :param raw_3d: 3D scan
    :param label_3d: 3D label
    :return: array of images (the 2D slices)
    """
    channels, width, height = raw_3d.shape
    raws = np.empty(shape=channels, dtype=object)
    labels = np.empty(shape=channels, dtype=object)
    for i in range(channels):
        raws[i] = Image.fromarray(raw_3d[i], mode='L')
        labels[i] = Image.fromarray(label_3d[i], mode='L')

    return raws, labels


def rand_rotate(raws: np.ndarray, labels: np.ndarray, max_abs_rot: float):
    """
    Rotate raw and label by a random angle. A joint transform.

    :param raws: the raw images
    :param labels: the corresponding labels
    :param max_abs_rot: max rotation in degrees
    """
    if max_abs_rot is None:
        return

    angle = 2 * (np.random.rand() - 0.5) * max_abs_rot
    for channel in range(len(raws)):
        raws[channel] = raws[channel].rotate(angle)
        labels[channel] = labels[channel].rotate(angle)


def rand_flip(raws: np.ndarray, labels: np.ndarray):
    """
    Randomly flip the images. A joint transform

    :param raws: the raw images
    :param labels: the corresponding labels
    """
    r = np.random.rand()
    if r < 0.25:
        return  # no flip
    elif r < 0.5:
        method = Image.FLIP_LEFT_RIGHT
    elif r < 0.75:
        method = Image.FLIP_TOP_BOTTOM
    else:
        method = Image.FLIP_LEFT_RIGHT | Image.FLIP_TOP_BOTTOM

    for channel in range(len(raws)):
        raws[channel] = raws[channel].transpose(method)
        labels[channel] = labels[channel].transpose(method)


def sharpen(raws: np.ndarray, labels: np.ndarray, factor: float):
    """
    Sharpen labels and raws by a given factor. A joint transform.

    :param raws: the raw images
    :param labels: the corresponding labels
    :param factor: <= 1 : less sharp, == 1 : copy, >= 1 : more sharp
    """
    if factor is None:
        return

    for channel in range(len(raws)):
        enhancer_raw = ImageEnhance.Sharpness(raws[channel])
        enhancer_label = ImageEnhance.Sharpness(labels[channel])
        raws[channel] = enhancer_raw.enhance(factor=factor)
        labels[channel] = enhancer_label.enhance(factor=factor)


def zoom(raws: np.ndarray, labels: np.ndarray, box_size: int):
    """
    Zoom on an area of the image, centered at the middle. A joint transform.

    :param raws: the raw images
    :param labels: the corresponding labels
    :param box_size: size (in pixels) of the square box centered at the middle, which defines the zoom area
    """
    if box_size is None:
        return

    w, h = raws[0].size  # images are square, w == h
    if box_size > w:
        raise ValueError(f'Received zoom box size {box_size}, which is bigger than image size ({w})')

    if box_size % 2 == 0:
        _box_size = box_size + 1
    else:
        _box_size = box_size

    img_center = (w // 2, h // 2)

    # crop the image, then resize (zoom) to original size (w, h)
    # define crop box
    x_left, x_right = img_center[0] - box_size // 2, img_center[0] + box_size // 2
    y_top, y_bottom = img_center[1] - box_size // 2, img_center[1] + box_size // 2

    for channel in range(len(raws)):
        raw_crop = raws[channel].crop((x_left, y_top, x_right, y_bottom))
        label_crop = labels[channel].crop((x_left, y_top, x_right, y_bottom))

        raws[channel] = raw_crop.resize((w, h), resample=Image.BICUBIC)
        labels[channel] = label_crop.resize((w, h), resample=Image.BICUBIC)


def contrast(raws: np.ndarray, factor: float):
    """
    Contrast raws by a given factor. A raw transform.

    :param raws: the raw images
    :param factor: <= 1 : less contrast, == 1 : copy, >= 1 : more contrast
    """
    if factor is None:
        return

    for channel in range(len(raws)):
        enhancer_raw = ImageEnhance.Contrast(raws[channel])
        raws[channel] = enhancer_raw.enhance(factor=factor)


def cluster(raws: np.ndarray, n_colors: int, kmeans: int = 0, method=0):
    """
    Cluster raws in clusters of colors. A raw transform.

    :param raws: the raw images
    :param n_colors: number of colors (clusters) to use, <= 256
    :param kmeans: convergence threshold, may be set to 0
    :param method: see https://pillow.readthedocs.io/en/stable/reference/Image.html#quantization-methods
    """
    if n_colors is None or kmeans is None or method is None:
        return

    for channel in range(len(raws)):
        raws[channel] = raws[channel].quantize(colors=n_colors, kmeans=kmeans, method=method).convert('L')


def _remove_irrelevant_label_clusters(labels, channel, img_center, dist_from_center_threshold):
    clustered_label, n_clusters = scim.label(labels[channel])
    # for each cluster, check if it is near the center
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = clustered_label == cluster_id
        cluster_coord = np.nonzero(cluster_mask)
        cluster_barycenter = np.mean(cluster_coord, axis=1)

        if np.linalg.norm(cluster_barycenter - img_center) < dist_from_center_threshold:
            labels[channel] = Image.fromarray(
                (255 * cluster_mask).astype(np.uint8), mode=labels[channel].mode)
            return True

    return False


def clean_label(labels, dist_from_center_thresh=10.):
    """
    Remove secondary aneurysms and stray white pixels in the labels. A label transform.

    :param labels: the labels to clean
    :param dist_from_center_thresh: threshold below which a point is considered 'at the center'
    of the image (in the sense of the L2-norm)
    """
    if dist_from_center_thresh is None:
        return

    w, h = labels[0].size
    img_center = (w // 2, h // 2)
    for channel in range(len(labels)):
        if np.max(labels[channel]) > 0:
            if not _remove_irrelevant_label_clusters(labels, channel, img_center, dist_from_center_thresh):
                labels[channel] = Image.fromarray(np.zeros_like(labels[channel]), mode=labels[channel].mode)


def _img_to_numpy(raws, labels):
    raws_np = np.empty(shape=(len(raws), *raws[0].size), dtype=np.float32)
    labels_np = np.empty(shape=(len(raws), *raws[0].size), dtype=np.float32)

    for channel in range(len(raws)):
        raws_np[channel] = np.asarray(raws[channel], dtype=np.float32)
        labels_np[channel] = np.asarray(labels[channel], dtype=np.float32)

    return raws_np, labels_np


def raw_transform(raws, contrast_factor, cluster_colors, cluster_kmeans=0, cluster_method=0):
    cluster(raws, cluster_colors, cluster_kmeans, cluster_method)
    contrast(raws, contrast_factor)
    return raws


def label_transform(labels, clean_dist_thresh=10.):
    clean_label(labels, clean_dist_thresh)
    return labels


def joint_transform(raws, labels, max_abs_rot, sharpen_factor, zoom_box, flip):
    sharpen(raws, labels, sharpen_factor)
    if flip:
        rand_flip(raws, labels)
    rand_rotate(raws, labels, max_abs_rot)
    zoom(raws, labels, zoom_box)
    # todo : dist transform ?
    raws_np, labels_np = _img_to_numpy(raws, labels)
    return raws_np, labels_np
