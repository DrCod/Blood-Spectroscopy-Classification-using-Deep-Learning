import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from numpy.fft import *
import argparse
from options import get_parser


hparams = get_parser().parse_args()

def spectral_collator(batch):
    
    x  = [el['x'] for el in batch]
    x = torch.tensor(x, dtype = torch.float)
    x = filter_signal(x)

    y  = [el['y'] for el in batch]

    y  = torch.tensor(y, dtype = torch.float)
        
    if hparams.model_type =="double":
        x_env = [el['x_env'] for el in batch]
        x_env = torch.tensor(x_env, dtype = torch.float)
        return x, x_env, y
    
    elif hparams.model_type =="single":
        return x, y
    
    
def test_spectral_collator(batch):
    
    x  = [el['x'] for el in batch]
    x = torch.tensor(x, dtype = torch.float)
    x = filter_signal(x)
    
    if hparams.model_type =="double":    

        x_env = [el['x_env'] for el in batch]
        x_env = torch.tensor(x_env, dtype = torch.float)

        return x, x_env
    
    elif hparams.model_type =="single":
        return x
    
    
def filter_signal(signal):
            
    sig = torch.fft.fft2(signal)
        
    bs = sig.shape[0]
    sig_dim = sig.shape[1]
    
    if not hparams.use_real_only:
        
        if hparams.use_threshold:
            arr = torch.zeros((bs, hparams.threshold, hparams.threshold, 1))

            for i in range(bs):
                arr[i, 1] = sig.real[i, :].unsqueeze(1)[:hparams.threshold]
                arr[i, 2] = sig.imag[i, :].unsqueeze(1)[:hparams.threshold]

            arr = arr.view(bs, -1)

            return arr
        
        else:
            arr = torch.zeros((bs, sig_dim, sig_dim, 1))
            
            for i in range(bs):
                arr[i, 1] = sig.real[i, :].unsqueeze(1)
                arr[i, 2] = sig.imag[i, :].unsqueeze(1)

            arr = arr.view(bs, -1)

            return arr
    else:
        return sig.real

class BloodDataset(Dataset):
    
    def __init__(self, features, env_features = None, targets = None, train_mode = True):
        
        self.train_mode = train_mode
        self.features = features
        
        self.env_features = env_features
        
        if train_mode:
            self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, item):
                
        x = self.features[item,:]
        
        if self.env_features:
            x_env = self.env_features[item, :]
                
            if self.train_mode:

                y = self.targets[item,:]

                return {

                    'x' : x,
                    'x_env' : x_env,
                    'y' : y
                }
            else:
                return {
                    'x' : x,
                    'x_env' : x_env
                }
        else:                
            if self.train_mode:

                y = self.targets[item,:]

                return {

                    'x' : x,
                    'y' : y
                }
            else:
                return {
                    'x' : x,
                }        