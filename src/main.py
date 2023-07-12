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
import argparse

### import Created library
from dataset import Create_dataset
from lerning_AE import Learning_AE
from lerning_VAE import Learning_VAE

#set the hyparameter 
# add argparse arguments
parser = argparse.ArgumentParser("Welcome to AE VAE code")
parser.add_argument("--patch_size", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--train_rate", type=float, default=None, help="EPOCH used for training")
parser.add_argument("--EPOCH", type=int, default=None, help="EPOCH used for training")
parser.add_argument("--z_dim", type=int, default=None, help="numlayer used for model setteing")
parser.add_argument("--input_size", type=int, default=None, help="mid_units used for model setteing")
parser.add_argument("--array_number", nargs='+', type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

### parameter
#specify batch size
# BATCH_SIZE = 100
BATCH_SIZE = args_cli.patch_size

#specify rate of train dataset and test dataset
# train_rate=0.8# use 80% for train
train_rate=args_cli.train_rate
#->test_rate=0.2 use 20% for test
# specify epochs
# num_epochs = 20
num_epochs = args_cli.EPOCH

###Create dataset
train_loader, val_loader = Create_dataset(BATCH_SIZE, train_rate)

### Main Function
#define variable
# z_dim = 2 #Dimentions of latent space
z_dim = args_cli.z_dim

# input_size = 28
input_size = args_cli.input_size

# array_number = [300, 100]
array_number = args_cli.array_number


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
