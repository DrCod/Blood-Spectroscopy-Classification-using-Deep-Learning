import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import logging
import sys, os
from tqdm import tqdm
import numpy as np
from options import get_parser

hparams = get_parser().parse_args()


# model train and validation utils
def train_fn(model, train_dataloader, criterion, optimizer, scheduler , device):
    
    logging.info("TRAIN")
    
    model.train()
    
    start_iter = 0
    final_loss = 0
    
    pbar = tqdm(iter(train_dataloader), leave = True, total = len(train_dataloader))
    
    for i, (data) in enumerate(pbar, start = start_iter):
        
        if hparams.model_type == "single":
            x, y = data
            inputs, targets = x.to(device), y.to(device)
            output = model(inputs)
        else:
            x, x_env, y = data 
            inputs , inputs_env, targets = x.to(device), x_env.to(device), y.to(device)
            output = model(inputs, inputs_env)
            
        optimizer.zero_grad()
        
        loss = criterion(output, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(train_dataloader)    
        
    return final_loss

def val_fn(model, valid_dataloader, criterion, device):
    
    logging.info("VALID")
    
    model.eval()
    
    final_loss = 0
    start_iter = 0
    valid_preds = []
    
    pbar= tqdm(iter(valid_dataloader), leave = True, total = len(valid_dataloader))
        
    
    for i, (data) in enumerate(pbar, start = start_iter):
        
        if hparams.model_type == "single":
            x, y = data
            inputs, targets = x.to(device), y.to(device)
            output = model(inputs)
        else:
            x, x_env, y = data 
            inputs , inputs_env, targets = x.to(device), x_env.to(device), y.to(device)
            output = model(inputs, inputs_env)
           
                
        loss = criterion(output, targets)
        
        final_loss += loss.item()
        
        valid_preds.append(output.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(valid_dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds


def inference_fn(model, test_dataloader, device):
    
    model.eval()
    
    preds = []

    pbar= tqdm(iter(test_dataloader), leave = True, total = len(test_dataloader))
        
    start_iter = 0
    
    for i, (data) in enumerate(pbar, start = start_iter):
        
        if hparams.model_type == "single":
            x = data
            inputs = x.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
        else:
            x, x_env= data 
            inputs , inputs_env = x.to(device), x_env.to(device)
        
            with torch.no_grad():
                outputs = model(inputs, inputs_env)
                
            
        preds.append(outputs.sigmoid().detach().cpu().numpy())
    
    preds = np.concatenate(preds)
    
    return preds

class SmoothBCEwLogits(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                                           self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss
    