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

class Encoder_AE(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(28*28, 300)#結合層(入力画像⇒300次元配列)
    self.lr2 = nn.Linear(300, 100)#結合層(300次元配列⇒100次元配列)
    self.lr3 = nn.Linear(100, z_dim)#結合層(100次元配列⇒2次元配列)
    self.relu = nn.ReLU()#活性化関数の層

  def forward(self, x):
    x = self.lr(x)#画像⇒300次元配列
    x = self.relu(x)#ReLu関数で活性化
    x = self.lr2(x)#300次元配列⇒100次元配列
    x = self.relu(x)#ReLu関数で活性化
    z = self.lr3(x)#100次元配列⇒潜在変数

    return z

class Decoder_AE(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(z_dim, 100)#結合層(潜在変数⇒100次元配列)
    self.lr2 = nn.Linear(100, 300)#結合層(100次元配列⇒300次元配列)
    self.lr3 = nn.Linear(300, 28*28)#結合層(300次元配列⇒復元画像)
    self.relu = nn.ReLU()#活性化関数の層

  def forward(self, z):
    x = self.lr(z)#潜在変数⇒100次元配列
    x = self.relu(x)#ReLu関数で活性化
    x = self.lr2(x)#100次元配列⇒300次元配列
    x = self.relu(x)#ReLu関数で活性化
    x = self.lr3(x)#300次元配列⇒復元画像
    x = torch.sigmoid(x)   #MNISTのピクセル値の分布はベルヌーイ分布に近いと考えられるので、シグモイド関数を適用します。
    return x

class AE(nn.Module):
  def __init__(self, z_dim):
    """
    #################################################################
    Variables:
      -x_in: 入力画像
      -x_out: 入力画像
      -z: 潜在変数
    #################################################################
    """
    super().__init__()
    self.encoder = Encoder_AE(z_dim)
    self.decoder = Decoder_AE(z_dim)

  def forward(self, x_in):
    z = self.encoder(x_in)#エンコーダ
    x_out = self.decoder(z)#デコーダ
    return x_out, z

# 損失関数
def criterion_AE(predict, target):
  """
  #################################################################
  Variables:
    -target:
    -predict:
  #################################################################
  """
  # 潜在ロス： クロスエントロピー
  loss = F.binary_cross_entropy(predict, target, reduction='sum')
  return loss