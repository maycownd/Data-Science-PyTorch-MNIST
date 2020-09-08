import sys
import numpy as np
sys.path.append("..")
# import standard PyTorch modules
from sklearn import metrics
import time
import pandas as pd
from tqdm import tqdm
import json
from IPython.display import clear_output
from models import ConvNet2
from preprocessing.fmnist import load_fmnist_torch, DICT_FASHION_MNIST
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter  # TensorBoard support


# calculate train time, writing train data to files etc.

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)  # On by default, leave it here for clarity

NUM_CLASS = 10
LR = 0.004
EPOCHS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH_MODEL = "../../Data/models/"


def save_model(model, path_model=PATH_MODEL, model_sufix="latest.pth"):
    torch.save(model.state_dict(),
               (path_model+model_sufix)
               )


def train_model(train_set,
                num_class=NUM_CLASS,
                device=DEVICE,
                lr=LR,
                epochs=EPOCHS,
                batch_size=64,
                criterion=nn.CrossEntropyLoss(),
                path_model=PATH_MODEL,
                verbose=True):
    """Train the model on train set.

    Args:
        train_set ([type]): [description]
        num_class ([type], optional): [description]. Defaults to NUM_CLASS.
        device ([type], optional): [description]. Defaults to DEVICE.
        lr ([type], optional): [description]. Defaults to LR.
        epochs ([type], optional): [description]. Defaults to EPOCHS.
        batch_size (int, optional): [description]. Defaults to 64.
        criterion ([type], optional): [description]. Defaults to nn.CrossEntropyLoss().
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    model = ConvNet2(num_class)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = criterion
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_id, (image, label) in enumerate(train_set):
            label, image = label.to(device), image.to(device)
            output = model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                if batch_id % 1000 == 0:
                    print(
                        '\nLoss :{:.4f} Epoch[{}/{}]\nTime: {} s\n'.format(
                            loss.item(),
                            epoch,
                            epochs,
                            (round((time.time() - start), 2))
                        )
                    )
    end = time.time()
    print("\nTime to complete (Training): {} s.".format(round(end-start), 2))
    save_model(model)
    torch.save(model.state_dict(),
               (path_model+"ConvNet_loss_"+str(round(loss.item(), 2))+".pth")
               )
    return model


def test_model(model, test_set, batch_size=64, device=DEVICE):
    """Test the trained model on Test Set.

    Args:
        model ([type]): [description]
        test_set ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 64.
        device ([type], optional): [description]. Defaults to DEVICE.
    """
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        model.eval()
        start = time.time()
        for image, label in test_set:
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            predicted = torch.argmax(outputs, dim=1)
            label = label.data.cpu().numpy()
            predicted = predicted.cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predicted)
    end = time.time()
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(
        labels_all,
        predict_all,
        target_names=DICT_FASHION_MNIST.values(),
        digits=2
    )
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    print('Test Accuracy of the model on the test images: {} %.\n\
           Time to complete (Testing): {} s.\n'.format(
        (100 * acc), round((end-start), 2)))
    print(confusion)
    print(report)
    return acc, report, confusion


def load_model(model_path,
               num_class=NUM_CLASS,
               device=DEVICE,
               saved_on_GPU=False,
               load_on_GPU=False,
               eval=True):
    """Load a saved model based on which device it was trained

    Args:
        model_path ([type]): [description]
        num_class ([type], optional): [description]. Defaults to NUM_CLASS.
        device ([type], optional): [description]. Defaults to DEVICE.
        saved_on_GPU (bool, optional): [description]. Defaults to False.
        load_on_GPU (bool, optional): [description]. Defaults to False.
        eval (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    model = ConvNet2(num_class)
    if saved_on_GPU:
        if load_on_GPU:
            model.load_state_dict(torch.load(model_path))
            model.to(device)
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        if load_on_GPU:
            model.load_state_dict(
                torch.load(model_path, map_location="cuda:0"))
            model.to(device)  # cuda
        else:
            model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()
    else:
        model.train()
    return model


def train_test():
    train, test, _, _ = load_fmnist_torch()
    # model_convnet = train_model(train)
    model_convnet = load_model(PATH_MODEL+"ConvNet_loss_0.09.pth", eval=True)
    print(type(model_convnet))
    test_model(model_convnet, test)


if __name__ == '__main__':
    train_test()
