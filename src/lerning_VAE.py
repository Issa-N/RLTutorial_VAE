# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from VAE import Encoder_VAE, Decoder_VAE, VAE
from VAE import criterion_VAE as criterion

def Learning_VAE(z_dim, num_epochs, train_loader,val_loader):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 使用するデバイス
  model = VAE(z_dim).to(device)#　学習モデル_AE
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#　最適化関数
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)#スケジューラ
  
  history_VAE = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels":[]}#学習結果保存用の配列
  
  for epoch in range(num_epochs):
    #訓練データの学習
    model.train()#モデルの選択
    for i, (x, labels) in enumerate(train_loader):
      input = x.to(device).view(-1, 28*28).to(torch.float32)
      output, z, ave, log_dev = model(input)
      #学習結果を保存
      history_VAE["ave"].append(ave)
      history_VAE["log_dev"].append(log_dev)
      history_VAE["z"].append(z)
      history_VAE["labels"].append(labels)
      loss = criterion(output, input, ave, log_dev)
      #最適化
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #途中過程の出力
      #if (i+1) % 50 == 0:
        #print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
        #損失関数の結果を保存
      history_VAE["train_loss"].append(loss)
    
    ##テストデータの学習
    model.eval()#モデルの選択
    with torch.no_grad():#メモリ削減のおまじない
      for i, (x, labels) in enumerate(val_loader):
        input = x.to(device).view(-1, 28*28).to(torch.float32)
        output, z, ave, log_dev = model(input)
        #損失関数および結果の保存
        loss = criterion(output, input, ave, log_dev)
        history_VAE["val_loss"].append(loss)
      #途中過程の出力
      print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')
    
  scheduler.step()
  np.save('save_history_VAE', history_VAE)
  torch.save(model, 'model_VAE.pth')
  return history_VAE