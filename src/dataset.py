# coding: utf-8
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

"""
#バッチサイズを指定
BATCH_SIZE = 100
#訓練データと試験データの比率
train_rate=0.8# 8割を訓練
test_rate=0.2# 2割をテスト
"""

def Create_dataset(BATCH_SIZE, train_rate):
	test_rate=1-train_rate

	#MNISTを読込
	trainval_data = MNIST("./data",
                   train=True,
                   download=True,
                   transform=transforms.ToTensor())
	#データセットの設定
	train_size = int(len(trainval_data) * 0.8)
	val_size = int(len(trainval_data) * 0.2)
	train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

	#訓練データセットを生成
	train_loader = DataLoader(dataset=train_data,#データセットの設定
                          batch_size=BATCH_SIZE,#バッチサイズの指定
                          shuffle=True,#抽出時にシャッフル
                          num_workers=0)# 並列実行数

	#試験データセットを生成
	val_loader = DataLoader(dataset=val_data,#データセットの設定
                        batch_size=BATCH_SIZE,#バッチサイズの指定
                        shuffle=True,#抽出時にシャッフル
                        num_workers=0)# 並列実行数

	#可視化
	print("##############################\n<dataset information>")
	print("\ttrain data size: ",len(train_data))   #train data size:  48000
	print("\ttrain iteration number: ",len(train_data)//BATCH_SIZE)   #train iteration number:  480
	print("\tval data size: ",len(val_data))   #val data size:  12000
	print("\tval iteration number: ",len(val_data)//BATCH_SIZE)   #val iteration number:  120
	print("##############################\n\n")


	return train_loader, val_loader
