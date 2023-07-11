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

### import Created library
from dataset import Create_dataset
from lerning_AE import Learning_AE
from lerning_VAE import Learning_VAE

### parameter
#�o�b�`�T�C�Y���w��
BATCH_SIZE = 100
#�P���f�[�^�Ǝ����f�[�^�̔䗦
train_rate=0.8# 8�����P��
#->test_rate=0.2# 2�����e�X�g
#�w�K�̃G�|�b�N��
num_epochs = 20

### ���C������
#�ϐ���`
z_dim = 2 #���ݕϐ��̎�����
#�w�K
print("\n<Start to learn by AE>")
history_AE=Learning_AE(z_dim, num_epochs, train_loader,val_loader)
print("\n<Start to learn by VAE>")
history_VAE=Learning_VAE(z_dim, num_epochs, train_loader,val_loader)
#�����̕ۑ�
with open('history_AE.pkl', 'wb') as f:
	pickle.dump(history_AE, f)
with open('history_VAE.pkl', 'wb') as f:
	pickle.dump(history_VAE, f)