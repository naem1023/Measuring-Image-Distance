import predictor.merger
from skimage import io
import torch
import h5py
import numpy as np

merger = predictor.merger.Merger('.\\model_state_dict.pth')
predict = merger.merge('.\\demo.jpg')
path_to_depth_v1 = 'Z:/nyu_data/nyu_depth_data_labeled.mat'
f = h5py.File(path_to_depth_v1)

# real_distance = io.imread('demo_depth.jpg')
# real_distance = real_distance[:, : ,0]
real_distance = depth = f['depths'][0]
real_distance = predictor.merger.transpose_img(real_distance)
rmse = predictor.merger.RMSELoss()
rmse_error = rmse(torch.tensor(predict), torch.tensor(real_distance))
print('rmse_error =', rmse_error.item())

predict_img = np.empty([predict.shape[0], predict.shape[1], 3])
predict_img[:, :, 0] = predict
predict_img[:, :, 1] = predict
predict_img[:, :, 2] = predict

io.imsave('demo_predict_distance.jpg', predict_img / 4)
