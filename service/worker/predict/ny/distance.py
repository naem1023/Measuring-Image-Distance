import torch
import predict.ny.ml_model
import numpy as np


class DistancePredictor:
    def __init__(self, model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = ny.ml_model.MIS()
        self.model.load_state_dict(state_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, image, point):
        self.model.eval()
        self.model = self.model.to(self.device)

        image = np.reshape(image, (1, 3, 640, 480))
        image = torch.tensor(image).to(self.device)

        point = [torch.tensor([point[0]]), torch.tensor([point[1]])]
        point = torch.tensor(point)
        point = torch.reshape(point, (2, 1))

        output = self.model((image, point))

        return output
