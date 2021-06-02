import predict.predictor.merger as pred_merger
from skimage import io
import torch
import h5py
import numpy as np
import h5py
import statistics
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm

def eval(ax, model_path, method='mid'):
    model_title = '-'.join(model_path.split('/')[-1].split('.')[0].split('-')[0:3])
    # Predict distance of all pixels
    merger = pred_merger.Merger(model_path)

    path_to_depth_v1 = '/home/relilau/nfs-home/nyu_data/nyu_depth_data_labeled.mat'
    f = h5py.File(path_to_depth_v1)

    rmse_list = []

    for idx, img in enumerate(tqdm(f['images'])):
        img_ = np.empty([480, 640, 3])
        img_[:, :, 0] = img[0, :, :].T
        img_[:, :, 1] = img[1, :, :].T
        img_[:, :, 2] = img[2, :, :].T
        img_ = img_.astype('uint8')
        io.imsave('demo.jpg', img_)

        predict = merger.merge('demo.jpg', method=method)

        # Transform depth array to rgb array
        predict_img = pred_merger.transform_to_rgb(predict)

        # Compare prediction and real distance
        real_distance = f['depths'][idx]
        real_distance = pred_merger.transpose_img(real_distance)
        rmse = pred_merger.RMSELoss()

        rmse_error = rmse(torch.tensor(predict), torch.tensor(real_distance))
        # print('rmse_error =', rmse_error.item())
        rmse_list.append(rmse_error.item())

        # io.imsave('demo_predict_distance.jpg', predict_img)

        # if idx == 1:
        #     break
    print(len(f['images']), idx)
    print(method, statistics.mean(rmse_list))

    # hist, bin_edges = np.histogram(rmse_list, bins=30, )
    N, bins, patches = ax.hist(rmse_list, bins=30, density=True, alpha=0.9, range=[0.0, 7.0], cumulative=False)

    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    ax.set_xlabel('x')
    ax.set_ylabel('Frequency')
    # ax.set_ylim(0, 2000)
    title = '%s, %s (mean=%.5f)' % (model_title, method, statistics.mean(rmse_list))
    file_name = 'test-%s.png' % method
    ax.set_title(title)
    # fig.savefig(file_name)
    # with open('merger-log.txt', 'a') as log_file:
    #     log_file.write(f'\n')
    # with open('predict-log.txt', 'a') as log_file:
    #     for val in rmse_list:
    #         log_file.write(f'{val} ')
    #     log_file.write('\n')

    return ax


if __name__ == '__main__':
    # with open('merger-log.txt', 'a') as log_file:
    #     log_file.write('='*10)
    # with open('predict-log.txt', 'a') as log_file:
    #     log_file.write('='*10)

    import time
    start = time.time()
    model_pathes = [
        # './predict/model_state_dict.pth',
        # './predict/mobilnetv3small-epoch-20-model-state-dict.pth',
        # './predict/mobilnetv3small-epoch-100-model-state-dict.pth',
        './predict/vgg11-epoch-100-model-state-dict.pth',
    ]
    methods = ['mid', 'mean', 'median', 'stdev']
    for model_path in model_pathes:
        model_title = '-'.join(model_path.split('/')[-1].split('.')[0].split('-')[0:3])
        fig, ax = plt.subplots(1, len(methods), figsize=(22, 6))

        all_rmse_list = []
        for idx, method in enumerate(methods):
            ax[idx] = eval(ax[idx], model_path, method=method)
        model_title = model_path.split('/')[-1].split('.')[0]
        fig.savefig(f'{model_title}-eval.png')
    end = time.time()

    print('elapsed time =', (end - start) / 60)


