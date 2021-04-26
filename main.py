import os
import train

data_path = 'Z:/nyu_data/nyu_depth_data_labeled.mat'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    # Get data from path and load to torchvision
    data = train.get_dataset(data_path)

    train.train(data)
