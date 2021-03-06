import h5py
import scipy.io
from torch.utils.data import Dataset, DataLoader
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class NyDataset(Dataset):
    """Newyork Data"""

    def __init__(self, root_dir, transform=None, x_point=10, y_point=10):
        """
        Args:
            root_dir (string):
                모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional):
                샘플에 적용될 Optional transform
            point (int):
                이미즈 한 변의 point 개수
        """
        self.root_dir = root_dir
        self.img_data_file = h5py.File(root_dir)
        self.transform = transform
        self.x_point = x_point
        self.y_point = y_point
        self.point = x_point * y_point

        f = h5py.File(self.root_dir)

        self.len = f['images'].shape[0]

        # flag for read image from *.mat or raw image.
        self.read_img = False
        self.read_depth = False
        self.X = [None] * self.__len__()
        self.depth = [None] * self.__len__()

    def __len__(self):
        return self.len * self.point

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        if type(idx) is list:
            converted_idx = np.array([(i // self.point, i % self.point) for i in idx])
        elif type(idx) is int:
            converted_idx = np.array([[idx // self.point, idx % self.point]])
            # converted_idx = np.reshape(converted_idx, (converted_idx.shape[0], 1))

        image = self.__get_image(self.root_dir, converted_idx[:, 0])
        # raw_depth_image = self.__get_raw_depth(self.root_dir, converted_idx[:, 0])

        if not self.read_depth:
            # Read idx depth image first time
            depth_image = self.__get_depth(self.root_dir, converted_idx[:, 0])
            depth_list, target_coordinate = self.get_depth_point(converted_idx[:, 1], depth_image=depth_image)
        else:
            # Already read idx dpeth image
            depth_list, target_coordinate = self.get_depth_point(converted_idx[:, 1])

        sample = {
            'image': image,
            'target_coordinate': target_coordinate
        }

        return sample, depth_list

    def get_depth_point(self, idxes, depth_image=None):
        if depth_image is None:
            return self.depth[idxes[0]]
        else:
            # Not coordinate of image, only order of training points.
            positions = [[idx % self.point // self.x_point, idx % self.point % self.x_point] for idx in idxes]
            x_interval = depth_image.shape[1] // self.x_point
            y_interval = depth_image.shape[2] // self.y_point

            depth = [depth_image[0][pos[0] * x_interval][pos[1] * y_interval] for pos in positions]
            target_coordinate = [[pos[0] * x_interval, pos[1] * y_interval] for pos in positions]

            depth = np.array(depth)

            self.depth[idxes[0]] = [depth, target_coordinate]

            return depth, target_coordinate

    def __get_raw_depth(self, root_dir, idx):
        rawDepth = self.img_data_file['rawDepths'][idx] / 4.0
        # return rawDepth
        # rawDepth_ = np.empty([480, 640, 3])
        # rawDepth_[:, :, 0] = rawDepth[:, :].T
        # rawDepth_[:, :, 1] = rawDepth[:, :].T
        # rawDepth_[:, :, 2] = rawDepth[:, :].T

        # image = io.imread(rawDepth_ / 4.0)
        return rawDepth

    def __get_depth(self, root_dir, idx):
        depth = self.img_data_file['depths'][idx]  # (1, 640, 480)
        # return depth
        # depth_ = np.empty([480, 640, 1])
        # depth_[:, :, 0] = depth[:, :].T
        # depth_[:, :, 1] = depth[:, :].T
        # depth_[:, :, 2] = depth[:, :].T
        # depth_ = depth.T

        transform_depth = depth.astype('float32') / 4.0
        return transform_depth

    def __get_image(self, root_dir, idx):
        if self.read_img:
            return self.X[idx[0]]
        else:
            img = self.img_data_file['images'][idx][0]  # (3, 640, 480)
            # return img
            # img_ = np.empty([480, 640, 3])
            # img_[:, :, 0] = img[0, :, :].T
            # img_[:, :, 1] = img[1, :, :].T
            # img_[:, :, 2] = img[2, :, :].T

            transform_img = img.astype('float32') / 255.0
            self.X[idx[0]] = transform_img

            return transform_img

