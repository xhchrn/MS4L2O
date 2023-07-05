'''
Please download the datasets from:
    - Ionosphere: https://archive.ics.uci.edu/dataset/52/ionosphere
    - Spambase: https://archive.ics.uci.edu/dataset/94/spambase
to the folder "./UCI_datasets/", uncompress the archives and then run this script.
'''

from os import listdir, makedirs
import random
import numpy as np
import scipy.io as sio
import pandas as pd


def normalize_data(data, axis):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


if __name__ == '__main__':
    # Process Ionosphere dataset first
    df = pd.read_csv('./UCI_datasets/ionosphere.data', header=None)
    w = np.expand_dims(np.array(df.iloc[:, :-1]), 0).astype(np.float32)
    w = normalize_data(w, axis=1)
    y = np.array([1 if label == 'g' else 0 for label in list(df.iloc[:,-1])]).reshape(1, df.shape[0], 1)
    generated_data_pth = './matdata/logistic-ionosphere-rho1e-1/0.mat'
    sio.savemat(generated_data_pth, {'W':w, 'Y':y, 'rho':1e-1})

    # Process Spambase dataset first
    df = pd.read_csv('./UCI_datasets/spambase.data', header=None)
    w = np.expand_dims(np.array(df.iloc[:, :-1]), 0).astype(np.float32)
    w = normalize_data(w, axis=1)
    y = np.array(df.iloc[:,-1]).reshape(1, df.shape[0], 1).astype(np.int32)
    generated_data_pth = './matdata/logistic-spambase-rho1e-1/0.mat'
    sio.savemat(generated_data_pth, {'W':w, 'Y':y, 'rho':1e-1})
