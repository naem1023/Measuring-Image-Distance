import numpy

import ny.distance as distance
from megadepth.MegaDepth.predictor import Predictor
from megadepth.MegaDepth.options.train_options import TrainOptions
from megadepth.MegaDepth.models.models import create_model

from skimage import io

class Merger:
    """Merge distance(meter) and depth information.
    """

    def __init__(self, distance_model_path, x_point = 10, y_point = 10):
        self.distance_predictor = self.get_distance_predictor(distance_model_path)
        self.depth_model = self.get_depth_model()
        self.x_point = x_point
        self.y_point = y_point

    def __read_image(self, img_path: str) -> numpy.ndarray:
        return io.imread(img_path)

    def merge(self, img_path: str) -> numpy.ndarray:
        """Calculate distance of all pixels using few distances and full depth.
        """
        img = self.__read_image(img_path).astype('float32') / 255.0
        distances = self.get_distance(img)
        depthes = self.get_depth(img_path)

        x_interval = img.shape[0] // self.x_point
        y_interval = img.shape[1] // self.y_point

        for distance_info in distances:
            point, predict_distance = distance_info
            predict_depth = img[point[0], point[1]]
            ratio = predict_distance / predict_depth
            x = point[0] - self.x_point
            y = point[1]

            for i in range(x_interval):
                for j in range(y_interval):
                    depthes[x + i, y + j] *= ratio

        return depthes

    def get_distance(self, img: numpy.ndarray) -> list:
        distances = []
        for i in range(self.x_point):
            target = [img.shape[0] // self.x_point * i, img.shape[1] // self.y_point * i]
            distances.append([target, self.distance_predictor.predict(img, target)])

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
