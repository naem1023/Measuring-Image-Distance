import predict.predictor.merger as pred_merger
from skimage import io
import torch
import h5py
import numpy as np
import h5py
import subprocess as sp

def compare():
    # Predict distance of all pixels
    merger = pred_merger.Merger('./predict/model_state_dict.pth')
    predict = merger.merge('demo.jpg')

    # Transform depth array to rgb array
    predict_img = pred_merger.transform_to_rgb(predict)

    # Compare prediction and real distance
    path_to_depth_v1 = '/home/relilau/nfs-home/nyu_data/nyu_depth_data_labeled.mat'
    f = h5py.File(path_to_depth_v1)
    real_distance = f['depths'][0]
    real_distance = pred_merger.transpose_img(real_distance)
    rmse = pred_merger.RMSELoss()

    rmse_error = rmse(torch.tensor(predict), torch.tensor(real_distance))
    print('rmse_error =', rmse_error.item())

    io.imsave('demo_predict_distance.jpg', predict_img)
def predict():
    # Predict distance of all pixels
    merger = pred_merger.Merger('./predict/model_state_dict.pth')
    predict = merger.merge('Vath_temp.jpg')

    # Transform depth array to rgb array
    predict_img = pred_merger.transform_to_rgb(predict)
    io.imsave('demo_predict_distance.jpg', predict_img)

if __name__ == '__main__':
    predict()
