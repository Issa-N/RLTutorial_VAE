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
#バッチサイズを指定
BATCH_SIZE = 100
#訓練データと試験データの比率
train_rate=0.8# 8割を訓練
#->test_rate=0.2# 2割をテスト
#学習のエポック数
num_epochs = 20

### メイン処理
#変数定義
z_dim = 2 #潜在変数の次元数
#学習
print("\n<Start to learn by AE>")
history_AE=Learning_AE(z_dim, num_epochs, train_loader,val_loader)
print("\n<Start to learn by VAE>")
history_VAE=Learning_VAE(z_dim, num_epochs, train_loader,val_loader)
#履歴の保存
with open('history_AE.pkl', 'wb') as f:
	pickle.dump(history_AE, f)
with open('history_VAE.pkl', 'wb') as f:
	pickle.dump(history_VAE, f)