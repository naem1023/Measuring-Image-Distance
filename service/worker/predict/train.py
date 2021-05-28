import ny.train
import ny.data

data_path = 'Z:/nyu_data/nyu_depth_data_labeled.mat'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    # Get data from path and load to torchvision
    dataset = ny.data.Data(data_path, test=True)
    train_data = dataset.get_dataset()

    trainer = ny.train.Train()
    trainer.train(train_data)
