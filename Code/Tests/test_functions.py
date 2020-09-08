import sys
sys.path.append("..")
from preprocessing.fmnist import load_fmnist_torch, DatasetFashionMNIST, load_sample
from training.pytorch_train import train_model, test_model, save_model, load_model
from training.models import ConvNet2
import torch
import requests
import json
checkdeploymentError = ""
train, test, train_set, test_set = load_fmnist_torch()


def checkdeployment():
    # function which tests whether the deployment is successful
    try:
        # Creating a sample data for fashion MNIST (28*28 images)
        headers = {'Content-Type': 'application/json'}

        resp = requests.post("http://localhost:5000/predict",
                             headers=headers,
                             files={
                                 "file": open('../../Data/tests/t-shirt.png',
                                              'rb')})
        print(resp)
        resp.json()
        result = (type(resp.json()[0]) == str)
    except ConnectionRefusedError:
        result = False
    return result


def checkTrainingMethod():
    try:
        train, _, = load_sample()
        model_convnet = train_model(train)
        result = True
    except RuntimeError:
        result = False
    return result


def checkModelSaving():
    try:
        train, _, = load_sample()
        model_convnet = train_model(train)
        save_model(model_convnet, path_model="../../Data/models/")
        result = True
    except IOError:
        result = False
    return result


def checkTrainingDataFormat():
    try:
        len_train, len_test = len(train_set), len(test_set)
        image, label = next(iter(train_set))

        result = len_train == 60000 == len_test == 10000 ==\
            isinstance(train_set, DatasetFashionMNIST) ==\
            isinstance(train, torch.utils.data.DataLoader) ==\
            isinstance(test_set, DatasetFashionMNIST) ==\
            image.shape[0] == 1 == image.shape[1] == 28 ==\
            image.shape[2] == 28
    except IOError:
        result = False
    return result


def checkAccuracy():
    conv_net = load_model(model_path="../../Data/models/", num_class=10)
    acc, cm, report = test_model(conv_net, test_set)
    result = acc >= 0.9
    return result
