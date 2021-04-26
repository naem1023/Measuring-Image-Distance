import skimage.io as io
import numpy as np
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

# v1,2_labeled: depths,images,rawDepths, rawDepthFilenames
path_to_depth_v2 = 'Z:/nyu_data/nyu_depth_v2_labeled.mat'
path_to_depth_v1 = 'Z:/nyu_data/nyu_depth_data_labeled.mat'
# v1_filenames: rawDepthFilenames, rawRgbFilenames
path_to_filename_v1 = 'Z:/nyu_data/nyu_depth_v1_filenames.mat'
f = h5py.File(path_to_depth_v2)


def get_ny_data():
    f = h5py.File(path_to_depth_v1)
    rawDepth = f['rawDepths']
    img = f['images']
    depth = f['depths']

    return img, depth, rawDepth


class NyDataset(Dataset):
    """Newyork Data"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.root_dir = root_dir
        self.img_data_file = h5py.File(root_dir)
        self.transform = transform

        f = h5py.File(self.root_dir)
        print('test', f['images'].shape)
        print('test', f['rawDepths'].shape)
        print('test', f['depths'].shape)

        self.len = f['images'].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.__get_image(self.root_dir, idx)
        raw_depth_image = self.__get_raw_depth(self.root_dir, idx)
        depth_image = self.__get_depth(self.root_dir, idx)

        # sample = {
        #     'image': image,
        #     'raw_depth_image': raw_depth_image,
        #     'depth_image': depth_image,
        # }
        #
        # if self.transform:
        #     sample = self.transform(sample)

        return image, depth_image

    def __get_raw_depth(self, root_dir, idx):
        rawDepth = self.img_data_file['rawDepths'][idx]

        # return rawDepth
        rawDepth_ = np.empty([480, 640, 3])
        rawDepth_[:, :, 0] = rawDepth[:, :].T
        rawDepth_[:, :, 1] = rawDepth[:, :].T
        rawDepth_[:, :, 2] = rawDepth[:, :].T

        # image = io.imread(rawDepth_ / 4.0)
        return rawDepth

    def __get_depth(self, root_dir, idx):
        depth = self.img_data_file['depths'][idx]
        # return depth
        depth_ = np.empty([480, 640, 3])
        depth_[:, :, 0] = depth[:, :].T
        depth_[:, :, 1] = depth[:, :].T
        depth_[:, :, 2] = depth[:, :].T

        # image = io.imread(depth_ / 4.0)
        return depth

    def __get_image(self, root_dir, idx):
        img = self.img_data_file['images'][idx]
        # return img
        img_ = np.empty([480, 640, 3])
        img_[:, :, 0] = img[0, :, :].T
        img_[:, :, 1] = img[1, :, :].T
        img_[:, :, 2] = img[2, :, :].T

        imag_ = img_.astype('float64')
        img = img.astype('float32')
        # image = io.imread(imag_ / 255.0)
        return img


# v1의 파일이름 배열 부르기
file_name = scipy.io.loadmat(path_to_filename_v1)
dname = file_name['rawDepthFilenames']
# print(dname)

rname = file_name['rawRgbFilenames']
# print(rname)

# 첫번째 rawDepths, 480*640
rawDepth = f['rawDepths'][0]
# print(f['rawDepths'].size)
print(f['rawDepths'])
# 전치
rawDepth_=np.empty([480,640,3])
rawDepth_[:,:,0]=rawDepth[:,:].T
rawDepth_[:,:,1]=rawDepth[:,:].T
rawDepth_[:,:,2]=rawDepth[:,:].T
print('rawDepth', type(rawDepth[0][0]))
#
# io.imshow(rawDepth_/4.0)
# io.show()

# 첫번째 image, 3*640*480
img = f['images'][0]

# #전치
img_=np.empty([480,640,3])
img_[:,:,0]=img[0,:,:].T
img_[:,:,1]=img[1,:,:].T
img_[:,:,2]=img[2,:,:].T

imag_=img_.astype('float32')
print('img', type(img[0][0][0]))
io.imshow(imag_/255.0)
# io.show()

# 대응하는 depth(image에 조절, 복원됨), 640*480
depth = f['depths'][0]


# #전치
depth_=np.empty([480,640,3])
depth_[:,:,0]=depth[:,:].T
depth_[:,:,1]=depth[:,:].T
depth_[:,:,2]=depth[:,:].T
print('depth', type(depth[0][0]))
#
# io.imshow(depth_/4.0)
# io.show()
