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
  #setting lerning enviroment
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# select device
  model = AE(z_dim).to(device)#learning model_AE
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#optimazation function
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)#scheduler
  
  #Define variable
  history_AE = {"val_loss": [], "train_loss": [], "z": [], "labels":[]}#array saved learning result
  
  #learning
  for epoch in range(num_epochs):
    #learning test data
    model.train()#select model
    for i, (x, labels) in enumerate(train_loader):
      input = x.to(device).view(-1, 28*28).to(torch.float32)
      output, z = model(input)
      #save result
      history_AE["z"].append(z)
      history_AE["labels"].append(labels)
      loss = criterion(output, input)
      #optimazation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #output halfway progress
      #if (i+1) % 50 == 0:
        #print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
      #save train loss
      history_AE["train_loss"].append(loss)
      
    ##Learning of test data
    model.eval()#select model
    with torch.no_grad():#memory reduction spell
      for i, (x, labels) in enumerate(val_loader):
        input = x.to(device).view(-1, 28*28).to(torch.float32)
        output, z = model(input)
        #save result
        loss = criterion(output, input)
        history_AE["val_loss"].append(loss)
      #output halfway progress
      print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')
    
  scheduler.step()
  np.save('save_history_AE', history_AE)
  torch.save(model, 'model_AE.pth')
  return history_AE