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
    raws = np.empty(shape=channels, dtype=Image)
    labels = np.empty(shape=channels, dtype=Image)
    for i in range(channels):
        raws[i] = Image.fromarray(raw_3d[i], mode='L')
        labels[i] = Image.fromarray(label_3d[i], mode='L')

    return raws, labels


def rand_rotate(raws: np.ndarray[Image], labels: np.ndarray[Image], max_abs_rot: float):
    """
    Rotate raw and label by a random angle. A joint transform.

    :param raws: the raw images
    :param labels: the corresponding labels
    :param max_abs_rot: max rotation in degrees
    """
    angle = 2 * (np.random.rand() - 0.5) * max_abs_rot
    for channel in range(len(raws)):
        raws[channel] = raws[channel].rotate(angle)
        labels[channel] = labels[channel].rotate(angle)


def sharpen(raws: np.ndarray[Image], labels: np.ndarray[Image], factor: float):
    """
    Sharpen labels and raws by a given factor. A joint transform.

    :param raws: the raw images
    :param labels: the corresponding labels
    :param factor: <= 1 : less sharp, == 1 : copy, >= 1 : more sharp
    """
    for channel in range(len(raws)):
        enhancer_raw = ImageEnhance.Sharpness(raws[channel])
        enhancer_label = ImageEnhance.Sharpness(labels[channel])
        raws[channel] = enhancer_raw.enhance(factor=factor)
        labels[channel] = enhancer_label.enhance(factor=factor)


def _shannon_zoom(img: Image, end_size: int):
    img_arr = np.array(img).astype(np.float64)
    img_size = len(img_arr)
    hl = end_size // 2 - img_size // 2  # half-length of the box to zoom into
    scaling = end_size / img_size

    # the fft of the end image is an image with the fft of the box on the middle, and the rest is zero-padded
    end_img_fft = np.zeros(shape=(end_size, end_size), dtype=np.complex)
    end_img_fft[hl:hl + img_size, hl:hl + img_size] = np.fft.fftshift(np.fft.fft2(img_arr))

    # this is a float64 image
    end_img = scaling**2 * np.real(np.fft.ifft2(np.fft.ifftshift(end_img_fft)))

    img_min, img_max = np.min(end_img), np.max(end_img)
    return Image.fromarray((255 * (end_img - img_min) / (img_max - img_min)).astype(np.uint8))


def zoom(raws: np.ndarray[Image], labels: np.ndarray[Image], box_size: int):
    """
    Zoom on an area of the image, centered at the middle. A joint transform.

    :param raws: the raw images
    :param labels: the corresponding labels
    :param box_size: size (in pixels) of the square box centered at the middle, which defines the zoom area
    """
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
    y_bottom, y_top = img_center[1] - box_size // 2, img_center[1] + box_size // 2

    for channel in range(len(raws)):
        raw_crop = raws[channel].crop((x_left, y_top, x_right, y_bottom))
        label_crop = labels[channel].crop((x_left, y_top, x_right, y_bottom))

        raws[channel] = _shannon_zoom(raw_crop, w)
        labels[channel] = _shannon_zoom(label_crop, w)


def contrast(raws: np.ndarray[Image], factor: float):
    """
    Contrast raws by a given factor. A raw transform.

    :param raws: the raw images
    :param factor: <= 1 : less contrast, == 1 : copy, >= 1 : more contrast
    """
    for channel in range(len(raws)):
        enhancer_raw = ImageEnhance.Contrast(raws[channel])
        raws[channel] = enhancer_raw.enhance(factor=factor)


def cluster(raws: np.ndarray[Image], n_colors: int, kmeans: int = 0, method=0):
    """
    Cluster raws in clusters of colors. A raw transform.

    :param raws: the raw images
    :param n_colors: number of colors (clusters) to use, <= 256
    :param kmeans: convergence threshold, may be set to 0
    :param method: see https://pillow.readthedocs.io/en/stable/reference/Image.html#quantization-methods
    """
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
    w, h = labels[0].size
    img_center = (w // 2, h // 2)
    for channel in range(len(labels)):
        if np.max(labels[channel]) > 0:
            if not _remove_irrelevant_label_clusters(labels, channel, img_center, dist_from_center_thresh):
                labels[channel] = Image.fromarray(np.zeros_like(labels[channel]), mode=labels[channel].mode)


def raw_transform(raws, contrast_factor, cluster_colors, cluster_kmeans=0, cluster_method=0):
    contrast(raws, contrast_factor)
    cluster(raws, cluster_colors, cluster_kmeans, cluster_method)
    return raws


def label_transform(labels, clean_dist_thresh=10.):
    clean_label(labels, clean_dist_thresh)
    return labels


def joint_transform(raws, labels, max_abs_rot, sharpen_factor, zoom_box):
    sharpen(raws, labels, sharpen_factor)
    rand_rotate(raws, labels, max_abs_rot)
    zoom(raws, labels, zoom_box)
    # todo : dist transform ?
    return raws, labels
