import predictor.merger
from skimage import io
import torch

merger = predictor.merger.Merger('.\\model_state_dict.pth')
predict = merger.merge('.\\demo.jpg')

real_distance = io.imread('demo_depth.jpg')

real_distance = real_distance[:, : ,0]

rmse = predictor.merger.RMSELoss()
rmse_error = rmse(torch.tensor(predict), torch.tensor(real_distance))
print('rmse_error =', rmse_error.item())
io.imsave('demo_predict_distance.jpg', predict)
