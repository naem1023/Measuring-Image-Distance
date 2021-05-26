import sys

from skimage import io

from options.train_options import TrainOptions
from predictor import Predictor

from models.models import create_model


opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

model = create_model(opt)
predictor = Predictor(model, 384, 512)

img_path = 'demo.jpg'
result = predictor.estimate_depth(img_path)
print(result.shape)
io.imsave('demo.png', result)
print("We are done")
sys.exit()

