import numpy
import torch
from torch.autograd import Variable
import numpy as np

from skimage import io
from skimage.transform import resize

# from models.HG_model import HGModel
from megadepth.MegaDepth.models.HG_model import HGModel


class Predictor:
    def __init__(self, model: HGModel, input_height: int = 384, input_width: int = 512):
        self.model = model
        self.input_height = input_height
        self.input_width = input_width

    def _read_image(self, image_path: str) -> numpy.ndarray:
        return io.imread(image_path)

    def estimate_depth(self, image_path: str) -> numpy.ndarray:
        self.model.switch_to_eval()

        img = np.float32(self._read_image(image_path)) / 255.0
        img = resize(img, (self.input_height, self.input_width), order=1)
        input_img = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda())
        pred_log_depth = self.model.netG.forward(input_images)
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)

        pred_inv_depth = 1 / pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)

        return pred_inv_depth
