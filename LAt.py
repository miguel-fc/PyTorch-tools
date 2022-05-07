import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class LAutoencoder(nn.Module):
    
    def __init__(self,encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(fc2_input_dim * fc2_input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12,encoded_space_dim))
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, fc2_input_dim * fc2_input_dim),
            nn.Sigmoid() )
        
    def forward(self, x):
            latent = self.encoder(x)
            x = self.decoder(latent)
            return x,latent

    ### Training function
    def train_epoch(self,model, device, dataloader, loss_fn, optimizer):
        model.train()
        train_loss = []
        for data in dataloader: 
            img = data
            img = img.view(img.size(0), -1).to(device)  
            # Encode and Decode data
            output,latent = model(img)
            loss = loss_fn(output, img)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch loss
            # print('\t train loss per batch: %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    ### Testing function
    def test_epoch(self, model, device, dataloader, loss_fn):
        model.eval()
        with torch.no_grad(): 
            conc_out = []
            conc_label = []
            for  data in dataloader:
                img = data
                img = img.view(img.size(0), -1).to(device) 
                # Encode and Decode data
                output, latent = model(img)
                # Append the network output and the original image to the lists
                conc_out.append(output.cpu())
                conc_label.append(img.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label) 
            # Evaluate global loss
            val_loss = loss_fn(conc_out, conc_label)
        return val_loss.data
