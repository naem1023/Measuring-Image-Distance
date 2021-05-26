import ny.distance
import h5py
import numpy as np
from skimage import io, transform

if __name__ == '__main__':
    model_path = 'model_state_dict.pth'
    predictor = ny.distance.DistancePredictor(model_path)

    path_to_depth_v1 = 'Z:/nyu_data/nyu_depth_data_labeled.mat'
    f = h5py.File(path_to_depth_v1)
    img = f['images'][0]
    transform_img = img.astype('float32') / 255.0
    point = [640 // 10 * 2, 480 // 10 * 2]

    output = predictor.predict(transform_img, point)
    print(output)
