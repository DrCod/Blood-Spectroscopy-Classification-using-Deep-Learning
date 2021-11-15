import torch.nn as nn
import torch
from models.model import *
from data_utils.dataloaders import *
from models.utils import *
from options import get_parser
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import sys
import time
import logging
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
import sys
import logging
from pprint import pprint
import json


def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def run_training(fold, folds, test_, feature_cols, tgt_cols, args, seed, feature_cols_env = None):
    
    seed_everything(seed)
    
    train_idx = folds[folds['kfold'] != fold].index
    valid_idx = folds[folds['kfold'] == fold].index
    
    train_df = folds.iloc[train_idx].reset_index(drop =True)
    valid_df = folds.iloc[valid_idx].reset_index(drop =True)
    
    x_train, y_train = train_df[feature_cols].values, train_df[tgt_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[tgt_cols].values
    x_test = test_[feature_cols].values
        
    if feature_cols_env:
        x_train_env, x_valid_env = train_df[feature_cols_env].values, valid_df[feature_cols_env].values
        x_test, x_test_env = test_[feature_cols].values, test_[feature_cols_env].values
        x_test_env = test_[feature_cols_env].values
    
    scaler = StandardScaler()
    
    scaler.fit(folds[feature_cols].values)
    
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test  = scaler.transform(x_test)
        
    if args.model_type != "single":
        train_dataset = BloodDataset(features =x_train, env_features=x_train_env, targets=y_train, train_mode= True)
        valid_dataset = BloodDataset(features =x_valid, env_features=x_valid_env, targets=y_valid, train_mode= True)
        testdataset   = BloodDataset(features = x_test, env_features=x_test_env,  targets = None,  train_mode = False)
    else:
        train_dataset = BloodDataset(features =x_train, env_features=None, targets=y_train, train_mode= True)
        valid_dataset = BloodDataset(features =x_valid, env_features=None, targets=y_valid, train_mode= True)
        testdataset   = BloodDataset(features = x_test, env_features=None,  targets = None,  train_mode = False)
        

    trainloader = DataLoader(
        train_dataset, collate_fn = spectral_collator, batch_size= args.BATCH_SIZE, shuffle=True)
    validloader = DataLoader(
        valid_dataset,collate_fn = spectral_collator, batch_size= args.BATCH_SIZE, shuffle=False)
    testloader = DataLoader(
        testdataset, collate_fn = test_spectral_collator, batch_size=args.BATCH_SIZE, shuffle=False)
    
    # Define feature space
    if args.use_real_only:
        num_features = len(feature_cols)
        
    elif args.use_threshold:
        num_features = args.threshold**2
    else:
        # use all features but it's going to be slow!
        num_features = len(feature_cols)**2
        
    if args.model_type == "single":
        model = SingleInputModel(
            hparams = args,
            num_features=num_features,
            num_targets= args.num_targets,
            hidden_size= args.hidden_size
            )
        
    elif args.model_type == "double":
        model = Model(
            hparams = args,
            num_features=num_features,
            num_env_features = args.num_env_features,
            num_targets= args.num_targets,
            hidden_size= args.hidden_size,
            hidden_size_env= args.hidden_size_env
        )
        
    else:
        RuntimeError("Invalid model type selected.")

    model.to(DEVICE)
    
    optimizer = optim.Adam(
        model.parameters(), lr= args.LR, weight_decay=args.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=args.EPOCHS, steps_per_epoch=len(trainloader))

    if not args.use_smoothing:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = SmoothBCEwLogits(smoothing= args.smoothing)

    oof = np.zeros((len(folds), len(tgt_cols)))
   
    early_stopping_steps = args.EARLY_STOPPING_STEPS
    early_step = 0
    
    min_loss = np.inf
    best_loss_epoch = -1
    
    for epoch in range(args.EPOCHS):
        
        logging.info(f"Epoch {epoch + 1}")
        
        #--------------------- TRAIN---------------------

        train_loss = train_fn(model, trainloader, criterion, optimizer, scheduler , DEVICE)
        
        #--------------------- VALID---------------------

        valid_loss, valid_preds = val_fn(model, validloader, criterion, DEVICE)
        
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_loss_epoch = epoch
            oof[valid_idx] = valid_preds
            
            torch.save(model.state_dict(), f"{args.model_output_folder}/SEED{seed}_FOLD{fold}_.pth")
            
        elif(args.EARLY_STOP):
            early_step += 1
            
            if(early_step >= early_stopping_steps):
                break
            
            
        if (epoch % 10 == 0)  or (epoch == (args.EPOCHS - 1)):
            print(f"Fold {fold}--Seed {seed}--Epoch {epoch}--Train Loss {train_loss:.6f}--Valid Loss {valid_loss:.6f}--Best Loss {min_loss:.6f}")

    
    #--------------------- PREDICTION---------------------

       
    if args.model_type == "single":
        model = SingleInputModel(
        hparams = args,
        num_features=num_features,
        num_targets= args.num_targets,
        hidden_size= args.hidden_size
        )
        
    elif args.model_type == "double":
        model = Model(
            hparams = args,
            num_features=num_features,
            num_env_features = args.num_env_features,
            num_targets= args.num_targets,
            hidden_size= args.hidden_size,
            hidden_size_env= args.hidden_size_env
        )
        
    else:
        RuntimeError("Invalid model type selected.")

    
    # Load the best model
    model.load_state_dict(torch.load(f"{args.model_output_folder}/SEED{seed}_FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Model Size: {num_params:,} trainable parameters")

    predictions = np.zeros((len(test_), len(tgt_cols)))
    predictions = inference_fn(model, testloader, DEVICE)

    return oof, predictions


def run_k_fold_double(train, test, feature_cols, feature_cols_env, tgt_cols, args, seed):
    
    oof = np.zeros((len(train), len(tgt_cols)))
    predictions = np.zeros((len(test), len(tgt_cols)))

    for fold in range(args.NFOLDS):
        oof_, pred_ = run_training(fold, train, test, feature_cols, tgt_cols, args, seed, feature_cols_env)

        predictions += pred_ / args.NFOLDS
        oof += oof_

    return oof, predictions

def run_k_fold_single(train, test, feature_cols, tgt_cols, args, seed):
    
    oof = np.zeros((len(train), len(tgt_cols)))
    predictions = np.zeros((len(test), len(tgt_cols)))

    for fold in range(args.NFOLDS):
        oof_, pred_ = run_training(fold, train, test,feature_cols, tgt_cols, args, seed)

        predictions += pred_ / args.NFOLDS
        oof += oof_

    return oof, predictions


def transform_labels_to_multilabel(df, old_cols, new_cols):
    
    for col in new_cols:
        name, status = col.split('_')[:-1], col.split('_')[-1]
        name = '_'.join(name)

        if status == 'ok':
            df.loc[:,col] = np.where(df.loc[:, name] == 'ok' , 1, 0)
        elif status == 'high':
            df.loc[:,col] = np.where(df.loc[:, name] == 'high' , 1, 0)
        elif status == 'low':
            df.loc[:,col] = np.where(df.loc[:, name] == 'low' , 1, 0)
            
    return df


def make_folds(df, tgt_cols, args):
    
    folds = df.copy()
    
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    mskf = MultilabelStratifiedKFold(n_splits = args.NFOLDS)

    for fold, (tr_idx, vl_idx) in enumerate(mskf.split(X = folds, y= folds[tgt_cols])):

        folds.loc[vl_idx, 'kfold'] = int(fold)

    folds['kfold'] = folds.kfold.astype(int)
    
    return folds


def main(args):
    
    global DEVICE
    
    pprint(args)
    
    with open(f'{args.model_output_folder}/{args.model_name}.json', 'w') as f:
        json.dump(vars(args), f)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train = pd.read_csv(f'data/{args.train_csv}')
    test_  = pd.read_csv(f'data/{args.test_csv}')
    

    feature_cols = [col for col in train.columns if "absorbance" in col]
    feature_cols_env = ['temperature' , 'humidity']

    target_cols = [
                    'hdl_cholesterol_human', 'hemoglobin(hgb)_human', 'cholesterol_ldl_human'
                ]
    
    new_cols = ['hdl_cholesterol_human_ok','hdl_cholesterol_human_high', 'hdl_cholesterol_human_low', 
                'cholesterol_ldl_human_ok', 'cholesterol_ldl_human_high', 'cholesterol_ldl_human_low',
               'hemoglobin(hgb)_human_ok', 'hemoglobin(hgb)_human_high', 'hemoglobin(hgb)_human_low'
               ]
    
    
    train_ = transform_labels_to_multilabel(train, target_cols, new_cols)
    # create folds
    folds = make_folds(train_, new_cols, args)

    # Averaging on multiple SEEDS
    
    SEED = [940, 1513, 1269] #1392, 1119]  # <-- Update
    
    oof = np.zeros((len(train), len(new_cols)))
    predictions = np.zeros((len(test_), len(new_cols)))

    for seed in SEED:

        if args.model_type =="double":
            
            oof_, predictions_ = run_k_fold_double(folds, test_, feature_cols, new_cols, args, seed, feature_cols_env)
            oof += oof_ / len(SEED)
            predictions += predictions_ / len(SEED)
            
        elif args.model_type == "single":
            oof_, predictions_ = run_k_fold_single(folds, test_, feature_cols, new_cols, args, seed)
            oof += oof_ / len(SEED)
            predictions += predictions_ / len(SEED)
            
            
    train[new_cols] = oof
    test_[new_cols] = predictions

    # save outputs
    train.to_csv(f'{args.model_output_folder}/oof_{args.model_name}.csv', index=False)
    test_.to_csv(f'{args.model_output_folder}/test_{args.model_name}.csv', index=False)
    
    print('Training complete!')
    
if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    