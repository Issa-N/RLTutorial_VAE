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
  def __init__(self, z_dim, input_size, array_number):
    super().__init__()
    self.lr = nn.Linear(input_size*input_size, array_number[0])#convolution layer(input image -> 300array)
    self.lr2 = nn.Linear(array_number[0], array_number[-1])#convolution layer(300 -> 100array)
    self.lr_ave = nn.Linear(array_number[-1], z_dim)#mean
    self.lr_dev = nn.Linear(array_number[-1], z_dim)#varient
    self.relu = nn.ReLU()#active function

  def forward(self, x):
    x = self.lr(x)#input image -> 300array
    x = self.relu(x)#ReLu
    x = self.lr2(x)#300->100arrat
    x = self.relu(x)#ReLu
    ave = self.lr_ave(x)#100array->mean(2D)
    log_dev = self.lr_dev(x)#100array->varient(2D)

    ep = torch.randn_like(ave)   #normal distribution(mu_0, sigma=1)
    z = ave + torch.exp(log_dev / 2) * ep   #latent varient
    return z, ave, log_dev

class Decoder_VAE(nn.Module):
  def __init__(self, z_dim, input_size, array_number):
    super().__init__()
    self.lr = nn.Linear(z_dim, array_number[-1])#convolution layer(latent variable -> 100array)
    self.lr2 = nn.Linear(array_number[-1], array_number[0])#convolution layer(100 -> 300array)
    self.lr3 = nn.Linear(array_number[0], input_size*input_size)#convolution layer(300array -> output image)
    self.relu = nn.ReLU()#active function

  def forward(self, z):
    x = self.lr(z)#latent variable->100array
    x = self.relu(x)#ReLu
    x = self.lr2(x)#100->300array
    x = self.relu(x)#ReLu
    x = self.lr3(x)#300array->output image
    x = torch.sigmoid(x)   #sigmoid
    return x

class VAE(nn.Module):
  def __init__(self, z_dim, input_size, array_number):
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
    self.encoder = Encoder_VAE(z_dim, input_size, array_number)
    self.decoder = Decoder_VAE(z_dim, input_size, array_number)

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
    -ave: mean
    -log_dev: varient(log value)
  #################################################################
  """
  # potential loss; Cross entropy
  bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
  # Reconstruct loss; (Average of BCE loss): E(w)=-1/N Sigma N(tlog(x)+(1-t)log(1-x))
  kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
  # reconstruction loss + potential loss
  loss = bce_loss + kl_loss
  return loss
