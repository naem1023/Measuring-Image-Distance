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

def eval(ax, method='mid'):
    # Predict distance of all pixels
    merger = pred_merger.Merger('./predict/model_state_dict.pth')

    path_to_depth_v1 = '/home/relilau/nfs-home/nyu_data/nyu_depth_data_labeled.mat'
    f = h5py.File(path_to_depth_v1)

    rmse_list = []

    for idx, img in enumerate(f['images']):
        img_ = np.empty([480, 640, 3])
        img_[:, :, 0] = img[0, :, :].T
        img_[:, :, 1] = img[1, :, :].T
        img_[:, :, 2] = img[2, :, :].T
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

        if idx > 2:
            break

        # io.imsave('demo_predict_distance.jpg', predict_img)
    print(len(f['images']), idx)
    print(method, statistics.mean(rmse_list))

    hist, bin_edges = np.histogram(rmse_list, bins=30, density=True, range=[0.0, 7.0])
    N, bins, patches = ax.hist(rmse_list, bin_edges, cumulative=False, color='#6495ED')

    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    ax.set_xlabel('x')
    ax.set_ylabel('Frequency')
    # ax.set_ylim(0, 2000)
    title = '%s (mean=%.5f)' % (method, statistics.mean(rmse_list))
    file_name = 'test-%s.png' % method
    ax.set_title(title)
    # fig.savefig(file_name)
    with open('merger-log.txt', 'a') as log_file:
        log_file.write(f'\n')
    with open('predict-log.txt', 'a') as log_file:
        for val in rmse_list:
            log_file.write(f'{val} ')
        log_file.write('\n')

    return ax


if __name__ == '__main__':
    with open('merger-log.txt', 'a') as log_file:
        log_file.write('='*10)
    with open('predict-log.txt', 'a') as log_file:
        log_file.write('='*10)

    methods = ['mid', 'mean', 'median', 'stdev']
    fig, ax = plt.subplots(1, len(methods), figsize=(20, 6))

    all_rmse_list = []
    for idx, method in enumerate(methods):
        ax[idx] = eval(ax[idx], method=method)
    fig.savefig('eval.png')


