
from utils.metrics import metric, test_metric

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pickle import dump, load

import warnings
warnings.filterwarnings('ignore')


def train(model, partition, optimizer, criterion, args):
    train_loader = DataLoader(partition['train'], batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)        
    
    model.train() # Turn on the evaluation mode    
    train_loss = 0.0
    train_metric = 0.0 
    amp_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    PATH = './scaler/'
    scaler = load(open(PATH+'scaler.pkl', 'rb'))

    for batch, data in enumerate(train_loader, 0):        
        optimizer.zero_grad()        
        # get the inputs
        X, y_true = data   
        X = X.transpose(0,1) # X [batch_size, X_seq_len, feature] -> [X_seq_len, batch_size, feature]
        X = X.cuda()
        y_true = y_true[:, :, -1].cuda()  
        
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            
            model.hidden = [hidden.cuda() for hidden in model.init_hidden()]
            y_pred = model(X)            
            loss = criterion(y_pred.to(torch.float32), y_true.to(torch.float32)) 
            
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)       

        train_loss += loss.item()
        y_pred = scaler.inverse_transform(y_pred.cpu().detach().numpy())
        y_true = scaler.inverse_transform(y_true.cpu().detach().numpy())
        train_metric += metric(y_pred, y_true)
        
    train_loss = train_loss / len(train_loader)
    train_metric = train_metric / len(train_loader)
    return model, train_loss, train_metric


def validate(model, partition, criterion, args):
    val_loader = DataLoader(partition['val'], batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)    
    model.eval() # Turn on the evaluation mode
    val_loss = 0.0 
    val_metric = 0.0
    PATH = './scaler/'
    scaler = load(open(PATH+'scaler.pkl', 'rb'))

    
    with torch.no_grad():
        for data in val_loader:            
            X, y_true = data
            X = X.transpose(0,1)
            X = X.cuda()          
            y_true = y_true[:, :, -1].cuda()         
            
            model.hidden = [hidden.cuda() for hidden in model.init_hidden()]
            y_pred = model(X)            
            loss = criterion(y_pred.view(-1), y_true.view(-1))
            y_pred = scaler.inverse_transform(y_pred.cpu().detach().numpy())
            y_true = scaler.inverse_transform(y_true.cpu().detach().numpy())
            val_metric += metric(y_pred, y_true)
        
        val_loss = val_loss / len(val_loader)
        val_metric = val_metric / len(val_loader)            
    return val_loss, val_metric


def test(model, partition, args, capacity):
    test_loader = DataLoader(partition['test'], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    model.eval() # Turn on the evaluation mode    
    preds = []
    trues = []
    PATH = './scaler/'
    scaler = load(open(PATH+'scaler.pkl', 'rb'))
    
    with torch.no_grad():
        for data in test_loader:            
            X, y_true = data
            X = X.transpose(0,1)
            X = X.cuda()            
            y_true = y_true[:, :, -1].cuda()  
            
            model.hidden = [hidden.cuda() for hidden in model.init_hidden()]
            y_pred = model(X)   
            
            preds.append(y_pred.detach().cpu().numpy())
            trues.append(y_true.detach().cpu().numpy())
            
        preds = np.array(preds)    
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-1], 1)
        trues = trues.reshape(-1, trues.shape[-1], 1)
        print('test shape:', preds.shape, trues.shape)
        preds = scaler.inverse_transform(preds)
        trues = scaler.inverse_transform(trues)
        
    mae, mse, Cv_rmse, kr_error, mape, mspe = test_metric(preds, trues, capacity)
    print('mse:{:2.4f}, mae:{:2.4f}, Cv_rmse:{:2.4f}, kr_error:{:2.4f}, mape:{:2.4f}, mspe:{:2.4f}'.format(mse, mae, Cv_rmse, kr_error, mape, mspe))            
    return Cv_rmse
    

def predict(model, partition, args):# RNN은 고정된 seq_len을 가지므로, 트랜스포머와 같이 batch_size=1로 적용불가
    test_loader = DataLoader(partition['test'], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model.eval() # Turn on the evaluation mode
    preds = []
    trues = []
    PATH = './scaler/'
    scaler = load(open(PATH+'scaler.pkl', 'rb'))
    
    with torch.no_grad():
        for data in test_loader:            
            X, y_true = data
            X = X.transpose(0,1)
            X = X.cuda()            
            y_true = y_true[:, :, -1].cuda()  
            
            model.hidden = [hidden.cuda() for hidden in model.init_hidden()]
            y_pred = model(X)     
            
            preds.append(y_pred.cpu().detach().numpy())
            trues.append(y_true.cpu().detach().numpy())        
            
        preds = np.array(preds) 
        trues = np.array(trues)
        print(preds.shape, trues.shape)            
        preds = preds.reshape(-1, preds.shape[-1], 1)
        trues = trues.reshape(-1, trues.shape[-1], 1)
        print(preds.shape, trues.shape)            
        preds = scaler.inverse_transform(preds)
        trues = scaler.inverse_transform(trues)                         
    return preds, trues