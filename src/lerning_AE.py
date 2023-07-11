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

from AE import Encoder_AE, Decoder_AE, AE
from AE import criterion_AE as criterion

def Learning_AE(z_dim, num_epochs, train_loader,val_loader):
  #学習の設定
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 使用するデバイス
  model = AE(z_dim).to(device)#　学習モデル_AE
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#　最適化関数
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)#スケジューラ
  
  #変数定義
  history_AE = {"val_loss": [], "train_loss": [], "z": [], "labels":[]}#学習結果保存用の配列
  
  #学習
  for epoch in range(num_epochs):
    #訓練データの学習
    model.train()#モデルの選択
    for i, (x, labels) in enumerate(train_loader):
      input = x.to(device).view(-1, 28*28).to(torch.float32)
      output, z = model(input)
      #学習結果を保存
      history_AE["z"].append(z)
      history_AE["labels"].append(labels)
      loss = criterion(output, input)
      #最適化
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #途中過程の出力
      #if (i+1) % 50 == 0:
        #print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
      #損失関数の結果を保存
      history_AE["train_loss"].append(loss)
      
    ##テストデータの学習
    model.eval()#モデルの選択
    with torch.no_grad():#メモリ削減のおまじない
      for i, (x, labels) in enumerate(val_loader):
        input = x.to(device).view(-1, 28*28).to(torch.float32)
        output, z = model(input)
        #損失関数および結果の保存
        loss = criterion(output, input)
        history_AE["val_loss"].append(loss)
      #途中過程の出力
      print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')
    
  scheduler.step()
  np.save('save_history_AE', history_AE)
  torch.save(model, 'model_AE.pth')
  return history_AE