import numpy as np
from utils.tools import StandardScaler


def metric(y_pred, y_true): # metric으로 Cv_rmse 사용 (향후 multivariate에서 사용)
#     y_pred = np.squeeze(y_pred, axis=1)
#     y_true = np.squeeze(y_true, axis=1)    
    # y_pred = y_scaler.inverse_transform(y_pred.cpu().detach().numpy())
    # y_true = y_scaler.inverse_transform(y_true.cpu().detach().numpy())
    # y_pred = y_pred.cpu().detach().numpy()
    # y_true = y_true.cpu().detach().numpy()
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)    
    mae = MAE(y_pred, y_true)      
    Cv_rmse = Cv_RMSE(y_pred, y_true)
    return Cv_rmse    

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def KR_ERROR(pred, true, capacity):
    indices = np.where(true > capacity*0.1)
    true_filtered = true[indices]
    pred_filtered = pred[indices]
    error = np.abs(true_filtered - pred_filtered)/capacity
    return np.mean(error)*100

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def Cv_RMSE(pred, true):
    return np.sqrt(MSE(pred, true))/np.mean(true)*100

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def test_metric(pred, true, capacity):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    Cv_rmse = Cv_RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)    
    kr_error = KR_ERROR(pred, true, capacity)
    return mae,mse,Cv_rmse,kr_error, mape,mspe