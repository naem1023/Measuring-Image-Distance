import ny.distance
import h5py
import numpy as np
from skimage import io, transform
import os

root_path = '.\\'
model_name = 'model_state_dict.pth'
model_path = os.path.join(root_path, model_name)
predictor = ny.distance.DistancePredictor(model_path)

path_to_depth_v1 = 'Z:/nyu_data/nyu_depth_data_labeled.mat'
f = h5py.File(path_to_depth_v1)
img = f['images'][0]
transform_img = img.astype('float32') / 255.0

depth = f['depths'][0]

print('shape =>', transform_img.shape)
for point in range(5):
    target = [640 // 10 * point, 480 // 10 * point]
    print(target)
    output = predictor.predict(transform_img, target)
    print('real distance =', depth[target[0]][target[1]], 'predict distance =', output.item() * 4)

    print()
