import numpy as np
from PIL import Image, ImageEnhance
from utils import get_raws, get_labels, load_data
from dataviz import view_sample, show_aneurysm
import scipy.ndimage as scim
import os
import h5py


class DataEnhancer:
    def __init__(self, data):
        self.data = []
        raws = get_raws(data)
        labels = get_labels(data)
        sample_shape = raws[0].shape
        n_channels = sample_shape[0]

        for raw, label in zip(raws, labels):
            self.data.append([])
            for i in range(n_channels):
                self.data[-1].append([Image.fromarray(raw[i], mode='L'),
                                      Image.fromarray(label[i], mode='L')])

        self.data_shape = (len(self.data), *sample_shape)
        self.img_size = self.data[0][0][0].size

    def raws(self, dtype):
        raws = np.empty(shape=self.data_shape, dtype=dtype)
        for i in range(self.data_shape[0]):
            for channel in range(self.data_shape[1]):
                raws[i][channel] = np.asarray(
                    self.data[i][channel][0], dtype=dtype)

        return raws

    def labels(self, dtype):
        labels = np.empty(shape=self.data_shape, dtype=dtype)
        for i in range(self.data_shape[0]):
            for channel in range(self.data_shape[1]):
                labels[i][channel] = np.asarray(
                    self.data[i][channel][1], dtype=dtype)

        return labels

    def rand_rotate(self, max_abs_rot):
        """
        Rotate all raws and labels by a random angle

        :param max_abs_rot: max rotation in degrees
        """
        for i in range(self.data_shape[0]):
            angle = 2 * (np.random.rand() - 0.5) * max_abs_rot

            for channel in range(self.data_shape[1]):
                raw, label = self.data[i][channel][0], self.data[i][channel][1]
                # todo : might have to use 'expand=True'
                self.data[i][channel][0] = raw.rotate(angle=angle, resample=Image.NEAREST)
                self.data[i][channel][1] = label.rotate(angle=angle, resample=Image.NEAREST)

    def sharpen(self, factor):
        """
        Sharpen labels and raws by a given factor

        :param factor: <= 1 : less sharp, == 1 : copy, >= 1 : more sharp
        """
        for i in range(self.data_shape[0]):
            for channel in range(self.data_shape[1]):
                raw, label = self.data[i][channel][0], self.data[i][channel][1]
                enhancer_raw = ImageEnhance.Sharpness(raw)
                enhancer_label = ImageEnhance.Sharpness(label)

                self.data[i][channel][0] = enhancer_raw.enhance(factor=factor)
                self.data[i][channel][1] = enhancer_label.enhance(
                    factor=factor)

    def crop(self, final_size: int):
        raws_np, labels_np = self.raws(np.float32), self.labels(np.float32)
        img_center = (self.img_size[0]//2, self.img_size[1]//2)
        top_left = (img_center[0] - final_size//2, img_center[1] - final_size//2)
        bot_right = (img_center[0] + final_size//2, img_center[1] + final_size//2)
        return raws_np[:, :, top_left[0]:bot_right[0], top_left[1]:bot_right[1]], labels_np[:, :, top_left[0]:bot_right[0], top_left[1]:bot_right[1]]

    def contrast_raws(self, factor):
        """
        Contrast raws by a given factor

        :param factor: <= 1 : less contrast, == 1 : copy, >= 1 : more contrast
        """
        for i in range(self.data_shape[0]):
            for channel in range(self.data_shape[1]):
                raw = self.data[i][channel][0]
                enhancer_raw = ImageEnhance.Contrast(raw)

                self.data[i][channel][0] = enhancer_raw.enhance(factor=factor)

    def cluster_raws(self, n_colors: int, kmeans: int = 0, method=0):
        """
        Cluster images in clusters of colors

        :param n_colors: number of colors (clusters) to use, <= 256
        :param kmeans: convergence threshold, may be set to 0
        :param method: see https://pillow.readthedocs.io/en/stable/reference/Image.html#quantization-methods
        """
        for i in range(self.data_shape[0]):
            for channel in range(self.data_shape[1]):
                raw = self.data[i][channel][0]

                self.data[i][channel][0] = raw.quantize(
                    colors=n_colors, kmeans=kmeans, method=method).convert('L')

    def _remove_irrelevant_label_clusters(self, i, channel, label, img_center, dist_from_center_threshold):
        clustered_label, n_clusters = scim.label(label)
        # for each cluster, check if it is near the center
        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = clustered_label == cluster_id
            cluster_coord = np.nonzero(cluster_mask)
            cluster_barycenter = np.mean(cluster_coord, axis=1)
            if np.linalg.norm(cluster_barycenter - img_center) < dist_from_center_threshold:
                self.data[i][channel][1] = Image.fromarray(
                    (255 * cluster_mask).astype(np.uint8), mode=label.mode)
                return True
        return False

    def clean_labels(self, dist_from_center_threshold=10.):
        """
        Remove secondary aneurysms and stray white pixels in the labels.

        :param dist_from_center_threshold: threshold below which a point is considered 'at the center'
        of the image (in the sense of the L2-norm)
        """
        img_center = np.array(
            [self.data_shape[2]//2, self.data_shape[3]//2], dtype=int)
        for i in range(self.data_shape[0]):
            for channel in range(self.data_shape[1]):
                label = self.data[i][channel][1]
                if np.max(label) > 0:
                    if not self._remove_irrelevant_label_clusters(
                            i, channel, label, img_center, dist_from_center_threshold
                    ):
                        # there are labels, but not at the center
                        self.data[i][channel][1] = Image.fromarray(
                            np.zeros_like(label), mode=label.mode)

    def _img_to_numpy(self, dtype):
        raws, labels = self.raws(dtype), self.labels(dtype)
        raws_np = np.empty(shape=(*raws.shape, *self.img_size), dtype=dtype)
        labels_np = np.empty(shape=(*raws.shape, *self.img_size), dtype=dtype)

        for channel in range(len(raws)):
            raws_np[channel] = np.asarray(raws[channel], dtype=dtype)
            labels_np[channel] = np.asarray(labels[channel], dtype=dtype)

        return raws_np, labels_np

    @staticmethod
    def save(all_raws, all_labels, path: str, index: int):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        print(all_raws.shape)

        for i in range(len(all_raws)):
            raws = all_raws[i]
            labels = all_labels[i]
            with h5py.File(os.path.join(path, f'scan_{i}_{index}.h5'), 'w') as f:
                f.create_dataset('raw', data=raws)
                f.create_dataset('label', data=labels)


def data_enhancer_test():
    # For debugging purposes
    data = load_data('challenge_dataset/')
    enhancer = DataEnhancer(data=data)
    # enhancer.clean_labels(dist_from_center_threshold=10.)  # clean_labels : OK
    enhancer.rand_rotate(max_abs_rot=20.)  # rand_rotate : OK
    enhancer.contrast_raws(factor=2.1)  # contrast_raws : OK
    # enhancer.cluster_raws(n_colors=3, kmeans=0)  # cluster_raws : OK
    enhancer.sharpen(factor=1.1)  # sharpen : OK
    raws_crop, labels_crop = enhancer.crop(final_size=32)
    view_sample(raws_crop, 0)
    show_aneurysm(raws_crop, labels_crop, 25)


if __name__ == '__main__':
    data_enhancer_test()
