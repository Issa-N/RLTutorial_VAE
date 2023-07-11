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

class Encoder_VAE(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(28*28, 300)#結合層(1層目)
    self.lr2 = nn.Linear(300, 100)#結合層(2層目)
    self.lr_ave = nn.Linear(100, z_dim)#平均値を求める層
    self.lr_dev = nn.Linear(100, z_dim)#分散を求める層
    self.relu = nn.ReLU()#活性化関数の層

  def forward(self, x):
    x = self.lr(x)#画像⇒300次元配列
    x = self.relu(x)#ReLu関数で活性化
    x = self.lr2(x)#300次元配列⇒100次元配列
    x = self.relu(x)#ReLu関数で活性化
    ave = self.lr_ave(x)#100次元配列⇒平均
    log_dev = self.lr_dev(x)#100次元配列⇒分散

    ep = torch.randn_like(ave)   #平均0分散1の正規分布に従い生成されるz_dim次元の乱数
    z = ave + torch.exp(log_dev / 2) * ep   #再パラメータ化トリック
    return z, ave, log_dev

class Decoder_VAE(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(z_dim, 100)#結合層(潜在変数⇒100次元配列)
    self.lr2 = nn.Linear(100, 300)#結合層(300次元配列⇒100次元配列)
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

class VAE(nn.Module):
  def __init__(self, z_dim):
    """
    #################################################################
    Variables:
      -x_in: 入力画像
      -x_out: 入力画像
      -z: 潜在変数
      -ave: 平均値
      -log_dev: 分散(対数値)
    #################################################################
    """
    super().__init__()
    self.encoder = Encoder_VAE(z_dim)
    self.decoder = Decoder_VAE(z_dim)

  def forward(self, x_in):
    z, ave, log_dev = self.encoder(x_in)#エンコーダ
    x_out = self.decoder(z)#デコーダ
    return x_out, z, ave, log_dev

# 損失関数
def criterion_VAE(predict, target, ave, log_dev):
  """
  #################################################################
  Variables:
    -target:
    -predict:
    -ave: 平均値
    -log_dev: 分散(対数値)
  #################################################################
  """
  # 潜在ロス： クロスエントロピー
  bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
  # 再構築ロス（BCE誤差の平均値）：E(w)=-1/NΣN(tlog(x)+(1-t)log(1-x))
  kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
  # 再構成ロス + 潜在ロス
  loss = bce_loss + kl_loss
  return loss
