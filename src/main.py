# coding: utf-8
###import library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

### import Created library
from dataset import Create_dataset
from lerning_AE import Learning_AE
from lerning_VAE import Learning_VAE

### parameter
#specify batch size
BATCH_SIZE = 100
#specify rate of train dataset and test dataset
train_rate=0.8# use 80% for train
#->test_rate=0.2 use 20% for test
#specify epochs
num_epochs = 20

###Create dataset
train_loader, val_loader = Create_dataset(BATCH_SIZE, train_rate)

### Main Function
#define variable
z_dim = 2 #Dimentions of latent space
#Learning network
print("\n<Start to learn by AE>")
history_AE=Learning_AE(z_dim, num_epochs, train_loader,val_loader)
print("\n<Start to learn by VAE>")
history_VAE=Learning_VAE(z_dim, num_epochs, train_loader,val_loader)
#save learning result
with open('history_AE.pkl', 'wb') as f:
	pickle.dump(history_AE, f)
with open('history_VAE.pkl', 'wb') as f:
	pickle.dump(history_VAE, f)
