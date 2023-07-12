import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_fig():
    #Cread history dictonary file
    with open('history_AE.pkl', 'rb') as f:
        history_AE=pickle.load(f)
    with open('history_VAE.pkl', 'rb') as f:
        history_VAE=pickle.load(f)
    return history_AE,history_VAE

def plot_lossFig(history_AE,history_VAE):
    fig = plt.figure(figsize=(10,6))
    plt.suptitle('Learning Results and Losses(AE & VAE)',fontsize=20)

    train_loss_tensor = torch.stack(history_AE["train_loss"])
    train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(train_loss_np)
    ax1.set_ylabel("Loss")
    ax1.set_title("AE_Train", loc='left')

    val_loss_tensor = torch.stack(history_AE["val_loss"])
    val_loss_np = val_loss_tensor.to('cpu').detach().numpy().copy()
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(val_loss_np)
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Batch Number")
    ax2.set_title("AE_Test", loc='left')

    train_loss_tensor = torch.stack(history_VAE["train_loss"])
    train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.plot(train_loss_np)
    #ax3.set_ylabel("Loss")
    ax3.set_title("VAE_Train", loc='left')

    val_loss_tensor = torch.stack(history_VAE["val_loss"])
    val_loss_np = val_loss_tensor.to('cpu').detach().numpy().copy()
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(val_loss_np)
    #ax4.set_ylabel("Loss")
    ax4.set_xlabel("Batch Number")
    ax4.set_title("VAE_Test", loc='left')
    plt.savefig("lossFig.png")
    plt.close()

#plot latent space
def Create_latentFig(history_AE,history_VAE):
    #change variable type
    ##AE
    z_tensor_AE = torch.stack(history_AE["z"])
    labels_tensor_AE = torch.stack(history_AE["labels"])
    z_np_AE = z_tensor_AE.to('cpu').detach().numpy().copy()
    labels_np_AE = labels_tensor_AE.to('cpu').detach().numpy().copy()
    ##VAE
    ave_tensor_VAE = torch.stack(history_VAE["ave"])
    log_var_tensor_VAE = torch.stack(history_VAE["log_dev"])
    z_tensor_VAE = torch.stack(history_VAE["z"])
    labels_tensor_VAE = torch.stack(history_VAE["labels"])
    ave_np_VAE = ave_tensor_VAE.to('cpu').detach().numpy().copy()
    log_var_np_VAE = log_var_tensor_VAE.to('cpu').detach().numpy().copy()
    z_np_VAE = z_tensor_VAE.to('cpu').detach().numpy().copy()
    labels_np_VAE = labels_tensor_VAE.to('cpu').detach().numpy().copy()

    #plot figure(early)
    #Specify figure color
    map_keyword = "Set1"
    cmap = plt.get_cmap(map_keyword)

    #specify bachnumber
    batch_range=10
    batch_num =batch_range

    #figure setting
    fig=plt.figure(figsize=[10,5])
    plt.suptitle('Latent Variable Space(AE & VAE)',fontsize=20)

    train_loss_tensor = torch.stack(history_AE["train_loss"])
    train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
    ax1 = fig.add_subplot(1, 2, 1)
    for label in range(10):
        x = z_np_AE[:batch_num,:,0][labels_np_AE[:batch_num,:] == label]
        y = z_np_AE[:batch_num,:,1][labels_np_AE[:batch_num,:] == label]
        ax1.scatter(x, y, color=cmap(label/9), label=label, s=15)
        #ax1.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
        ax1.set_xlabel("AE",fontsize=20)
    ax1.legend(loc="upper left")
    ax2 = fig.add_subplot(1, 2, 2)
    for label in range(10):
        x = z_np_VAE[:batch_num,:,0][labels_np_VAE[:batch_num,:] == label]
        y = z_np_VAE[:batch_num,:,1][labels_np_VAE[:batch_num,:] == label]
        ax2.scatter(x, y, color=cmap(label/9), label=label, s=15)
        #ax2.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
        ax2.set_xlabel("VAE",fontsize=20)
    ax2.legend(loc="upper left")
    plt.savefig("latentSpaceEarly.png")
    plt.close()

    #plot figure(late)
    #specify bachnumber
    batch_num =len(z_np_VAE)-batch_range
    #figure setting
    fig=plt.figure(figsize=[10,5])
    plt.suptitle('Latent Variable Space(AE & VAE)',fontsize=20)

    train_loss_tensor = torch.stack(history_AE["train_loss"])
    train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
    ax1 = fig.add_subplot(1, 2, 1)
    for label in range(10):
        x = z_np_AE[batch_num:,:,0][labels_np_AE[batch_num:,:] == label]
        y = z_np_AE[batch_num:,:,1][labels_np_AE[batch_num:,:] == label]
        ax1.scatter(x, y, color=cmap(label/9), label=label, s=15)
        #ax1.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
        ax1.set_xlabel("AE",fontsize=20)
    ax1.legend(loc="upper left")
    ax2 = fig.add_subplot(1, 2, 2)
    for label in range(10):
        x = z_np_VAE[batch_num:,:,0][labels_np_VAE[batch_num:,:] == label]
        y = z_np_VAE[batch_num:,:,1][labels_np_VAE[batch_num:,:] == label]
        ax2.scatter(x, y, color=cmap(label/9), label=label, s=15)
        #ax2.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
        ax2.set_xlabel("VAE",fontsize=20)
    ax2.legend(loc="upper left")
    plt.savefig("latentSpaceLate.png")
    plt.close()




#Main
history_AE,history_VAE=draw_fig()
plot_lossFig(history_AE,history_VAE)
Create_latentFig(history_AE,history_VAE)