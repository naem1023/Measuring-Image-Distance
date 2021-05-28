import predictor.merger

from skimage import io
import torch
import h5py
import numpy as np


# Predict distance of all pixels
merger = predictor.merger.Merger('model_state_dict.pth')
predict = merger.merge('demo.jpg')

# Transform depth array to rgb array
predict_img = predictor.merger.transform_to_rgb(predict)

# Compare prediction and real distance
path_to_depth_v1 = 'Z:/nyu_data/nyu_depth_data_labeled.mat'
f = h5py.File(path_to_depth_v1)

real_distance = depth = f['depths'][0]
real_distance = predictor.merger.transpose_img(real_distance)
rmse = predictor.merger.RMSELoss()

rmse_error = rmse(torch.tensor(predict), torch.tensor(real_distance))
print('rmse_error =', rmse_error.item())

io.imsave('demo_predict_distance.jpg', predict_img)
