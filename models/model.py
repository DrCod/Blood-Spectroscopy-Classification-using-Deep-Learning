import torch
import torch.nn as nn
import torch.nn.functional as F

class CReLU(nn.Module):
    
    """CReLU Activation
     This is a modification of the classical CReLU activation proposed in this paper (https://arxiv.org/pdf/1603.05201.pdf)
    returns : CONCAT(relu(x), relu(-x))
    """
    def __init__(self):        
        super(CReLU, self).__init__()
    def forward(self,x):
        return torch.cat((F.relu(x), -F.relu(-x)), dim =1)

class SingleInputModel(nn.Module):
    def __init__(self, hparams, num_features, num_targets, hidden_size):
        super(SingleInputModel, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(
            nn.Linear(num_features, hparams.hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hparams.hidden_size*2)
        self.dropout2 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(hparams.hidden_size*2, hparams.hidden_size)

        self.batch_norm3 = nn.BatchNorm1d(hparams.hidden_size*2)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hparams.hidden_size*2, hparams.num_targets))
        self.crelu = CReLU()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.crelu(self.dense1(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.crelu(self.dense2(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x
    

class Model(nn.Module):

    def __init__(self, hparams, num_features, num_env_features, num_targets, hidden_size, hidden_size_env):
        super(Model, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.batch_norm_env = nn.BatchNorm1d(hparams.num_env_features)
        self.dense_env      = nn.utils.weight_norm(nn.Linear(hparams.num_env_features, hparams.hidden_size_env))
        self.dense1 = nn.utils.weight_norm(
            nn.Linear(num_features, hparams.hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hparams.hidden_size*2)
        self.dropout2 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(hparams.hidden_size*2, hparams.hidden_size)

        self.batch_norm3 = nn.BatchNorm1d((hparams.hidden_size)*2 + hparams.hidden_size_env)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear((hparams.hidden_size)*2 + hparams.hidden_size_env, hparams.num_targets))
        self.crelu = CReLU()

    def forward(self, x, x_env):
        x = self.batch_norm1(x)
        x = self.crelu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.crelu(self.dense2(x))
        
        x_env = self.batch_norm_env(x_env)
        x_env = self.dense_env(x_env)
        
        x = torch.cat((x, x_env), dim = 1)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x