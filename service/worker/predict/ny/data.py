import torch
import ny.ny


class Data:
    def __init__(self, path, test=False):
        self.test = test
        self.ny_dataset = ny.ny.NyDataset(path)

    def get_dataset(self, train_ratio=0.8):
        if self.test:
            train_len = 10
            test_len = 2
            train_dataset, test_dataset, _ = torch.utils.data.random_split(self.ny_dataset,
                                                                        [train_len, test_len,
                                                                         len(self.ny_dataset) - train_len - test_len])

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
        else:
            # Set split length
            train_len = int(len(self.ny_dataset) * train_ratio)
            test_len = len(self.ny_dataset) - train_len

            train_dataset, test_dataset = torch.utils.data.random_split(self.ny_dataset, [train_len, test_len])

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

        return train_loader, test_loader
