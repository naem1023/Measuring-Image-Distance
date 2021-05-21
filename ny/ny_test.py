import h5py
import numpy as np
from skimage import io, transform

# v1,2_labeled: depths,images,rawDepths, rawDepthFilenames
path_to_depth_v2 = 'Z:/nyu_data/nyu_depth_v2_labeled.mat'
path_to_depth_v1 = 'Z:/nyu_data/nyu_depth_data_labeled.mat'
# v1_filenames: rawDepthFilenames, rawRgbFilenames
path_to_filename_v1 = 'Z:/nyu_data/nyu_depth_v1_filenames.mat'
f = h5py.File(path_to_depth_v2)

# v1의 파일이름 배열 부르기
file_name = scipy.io.loadmat(path_to_filename_v1)
dname = file_name['rawDepthFilenames']

rname = file_name['rawRgbFilenames']

# 첫번째 rawDepths, 480*640
rawDepth = f['rawDepths'][0]
# 전치
rawDepth_=np.empty([480,640,3])
rawDepth_[:,:,0]=rawDepth[:,:].T
rawDepth_[:,:,1]=rawDepth[:,:].T
rawDepth_[:,:,2]=rawDepth[:,:].T
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

from tqdm import tqdm

def check_minmax():
    max_val = []
    min_val = []
    for origin_depth in tqdm(f['depths']):
        max_val.append(np.max(origin_depth))
        min_val.append(np.min(origin_depth))
        # depth_ = np.empty([480, 640, 1])
        # depth_[:, :, 0] = origin_depth[:, :].T
        # depth_[:, :, 1] = origin_depth[:, :].T
        # depth_[:, :, 2] = origin_depth[:, :].T
        #
        # print('depth', type(origin_depth[0][0]))

    print(max_val, min_val)

    # #전치