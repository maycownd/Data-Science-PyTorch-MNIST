import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import inspect

PATH_TRAIN = "../../Data/raw/fashion-mnist_train.csv"
PATH_TEST = "../../Data/raw/fashion-mnist_test.csv"
PATH_SAMPLE = "../../Data/raw/sample.csv"
SIZE = 28
DICT_FASHION_MNIST = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}


class DatasetFashionMNIST(Dataset):
    """Fashion MNIST dataset using Pytorch class Dataset.

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, path, transform=None):
        """Method to initilaize variables."""
        self.fashion_MNIST = pd.read_csv(path).values
        self.transform = transform

        # first column is of labels.
        self.label = np.asarray(self.fashion_MNIST[:, 0])

        # Dimension of Images = 28 * 28 * 1.
        # where height = width = 28 and color_channels = 1.
        self.image = np.asarray(self.fashion_MNIST[:, 1:])\
            .reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.label[index]
        image = self.image[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.label)


def load_fmnist_torch(path_train=PATH_TRAIN,
                      path_test=PATH_TEST, batch_size=64, **kwargs):
    """Load Train and Test DatasetFashionMNIST Dataloader

    Args:
        path_train ([type], optional): [description]. Defaults to PATH_TRAIN.
        path_test ([type], optional): [description]. Defaults to PATH_TEST.
        batch_size (int, optional): [description]. Defaults to 64.

    Returns:
        [type]: [description]
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                            [0.0],
                            [1.0]
                        )
        ]
    )
    trainset = DatasetFashionMNIST(path=path_train, transform=transform)
    testset = DatasetFashionMNIST(path=path_test, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False, batch_size=batch_size)
    return trainloader, testloader, trainset, testset


def load_sample(path=PATH_SAMPLE):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                            [0.0],
                            [1.0]
                        )
        ]
    )
    trainset = DatasetFashionMNIST(path=path, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, shuffle=True, batch_size=1)
    return trainloader, trainset


def output_label(label, dict_mapping=DICT_FASHION_MNIST):
    """Convert the output to proper class:
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"

    Args:
        label ([type]): [description]

    Returns:
        [type]: [description]
    """
    output_mapping = dict_mapping
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


if __name__ == "__main__":
    train, test, train_set, test_set = load_fmnist_torch()
    print(len(train_set))
    t_train = type(train_set)
    tt_train = type(train)
    print(t_train)
    print(tt_train)
    print(isinstance(train_set, DatasetFashionMNIST))
    print(isinstance(train, torch.utils.data.DataLoader))

    image, label = next(iter(train_set))
    print(image.shape[0]==1, image.shape[1]==28, image.shape[2]==28)
    plt.imshow(image.squeeze(), cmap="gray")
    print(output_label(label))

