import numpy

# import predict.ny.distance as distance
import predict.ny.distance as distance
import megadepth.MegaDepth.models
from megadepth.MegaDepth.predictor import Predictor
from megadepth.MegaDepth.options.train_options import TrainOptions
from megadepth.MegaDepth.models.models import create_model

from skimage import io
import torch
import torch.nn as nn
import numpy as np


def transform_to_rgb(depth: numpy.ndarray) -> numpy.ndarray:
    predict_img = np.empty([depth.shape[0], depth.shape[1], 3])
    predict_img[:, :, 0] = depth
    predict_img[:, :, 1] = depth
    predict_img[:, :, 2] = depth

    return predict_img


def transpose_img(img: numpy.ndarray, to_normal=False) -> numpy.ndarray:
    if to_normal:
        if len(img.shape) == 3:
            img_ = np.empty((img.shape[0], img.shape[1], 3))
            img_ = img_.astype('float32')
            img_[:, :, 0] = img[0, :, :].T
            img_[:, :, 1] = img[1, :, :].T
            img_[:, :, 2] = img[2, :, :].T
        elif len(img.shape) == 2:
            img_ = img.T.astype('float32')

    else:
        if len(img.shape) == 3:
            img_ = np.empty((3, img.shape[1], img.shape[0]))
            img_ = img_.astype('float32')
            img_[0, :, :] = img[:, :, 0].T
            img_[1, :, :] = img[:, :, 1].T
            img_[2, :, :] = img[:, :, 2].T
        elif len(img.shape) == 2:
            img_ = img.T.astype('float32')

    return img_


class RMSELoss(nn.Module):
    """Calculate RMSE Loss for validating test data.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        """Calculate RMSE Loss from two vectors.
        Plus epsilon for preventing zero division.
        """
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class Merger:
    """Merge distance(meter) and depth information.
    """

    def __init__(self, distance_model_path, x_point=10, y_point=10):
        self.distance_predictor = self.get_distance_predictor(distance_model_path)
        self.depth_model = self.get_depth_model()
        self.x_point = x_point
        self.y_point = y_point

    def __read_image(self, img_path: str) -> numpy.ndarray:
        return io.imread(img_path)

    # def __transpose_img(self, img) -> numpy.ndarray:
    #     img_ = np.empty([img.shape[2], img.shape[1], img.shape[0]])
    #     img_ = img_.astype('float32')
    #     img_[:, :, 0] = img[0, :, :].T
    #     img_[:, :, 1] = img[1, :, :].T
    #     img_[:, :, 2] = img[2, :, :].T
    #
    #     return img_

    def merge(self, img_path: str, method='mid') -> numpy.ndarray:
        """Calculate distance of all pixels using few distances and full depth.
        """
        img = self.__read_image(img_path).astype('float32') / 255.0

        depthes = self.get_depth(img_path, img.shape[0], img.shape[1])
        depthes = transpose_img(depthes)
        img = transpose_img(img)
        distances = self.get_distance(img)

        # How many interval in each axes
        x_interval = img.shape[1] // self.x_point
        y_interval = img.shape[2] // self.y_point

        if method == 'mid':
            mid_pivot = len(distances) // 2
            point, predict_distance = distances[mid_pivot]
            predict_depth = depthes[point[0], point[1]]
            ratio = predict_distance / predict_depth
        elif method == 'mean':
            ratio = 0
            for distance in distances:
                point, predict_distance = distance
                predict_depth = depthes[point[0], point[1]]
                ratio += predict_distance / predict_depth
            ratio /= len(distances)
        elif method == 'median':
            import statistics
            ratio_list = []
            for distance in distances:
                point, predict_distance = distance
                predict_depth = depthes[point[0], point[1]]
                ratio_list.append(predict_distance / predict_depth)
            ratio = statistics.median(ratio_list)
        elif method == 'stdev':
            import statistics
            ratio_list = []
            for distance in distances:
                point, predict_distance = distance
                predict_depth = depthes[point[0], point[1]]
                ratio_list.append(predict_distance / predict_depth)
            ratio = statistics.stdev(ratio_list)

        with open('merger-log.txt', 'a') as log_file:
            log_file.write(f'{ratio} ')

        for distance_info in distances:
            point, predict_distance = distance_info
            # predict_depth = depthes[point[0], point[1]]
            # ratio = predict_distance / predict_depth

            # Position of root point in region is left and top.
            x = point[0]
            y = point[1]

            # Set region in (x, y) to (x + x_interval - 1, y + y_interval - 1)
            # and calculate distance of region's all pixel.
            for i in range(x_interval):
                for j in range(y_interval):
                    depthes[x + i, y + j] *= ratio

        return transpose_img(depthes) * 4

    def get_distance(self, img: numpy.ndarray) -> list:
        distances = []
        for i in range(self.x_point):
            for j in range(self.y_point):
                target = [img.shape[1] // self.x_point * i, img.shape[2] // self.y_point * j]
                distances.append([target, self.distance_predictor.predict(img, target).item()])

        return distances

    def get_depth(self, img_path: str, x: int, y: int) -> numpy.ndarray:
        depth_predictor = Predictor(self.depth_model, x, y)
        return depth_predictor.estimate_depth(img_path)

    def get_distance_predictor(self, path):
        return distance.DistancePredictor(path)

    def get_depth_model(self):
        opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
        model = create_model(opt)
        return model
