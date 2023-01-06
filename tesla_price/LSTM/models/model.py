import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_window, num_layers, batch_size, dropout, use_bn):
        super(LSTM, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.output_window = output_window
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn 
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers) # 인자 받는 순서 확인 
        self.hidden = self.init_hidden()
        self.regressor = self.make_regressor() # output dim을 여기서 받도록 설정 
        
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    
    def make_regressor(self):
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))        
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2)) # hid_dim을 절반으로 나눠줌
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim // 2, self.output_window)) # hid_dim -> out_dim으로 변경
        regressor = nn.Sequential(*layers) # sequential로 넣어서 MLP로 만듦 
        return regressor
    
    def forward(self, x):
#         print('x', x.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden) # 새로 업데이트된 lstm_out(한스텝에서의 output값)과 self.hidden (모둔 hidden state) return해줌 
#         print('output', lstm_out.shape, self.hidden[0].shape, self.hidden[1].shape)
#         print('hihi', lstm_out[-1].shape)
#         print('hihi2', (lstm_out[-1].view(self.batch_size, -1).shape))
        y_pred = self.regressor(lstm_out[-1].view(self.batch_size, -1)) # t스텝에서의 값을 regressor를 통해 예측         
#         print('pred', y_pred.shape)
        return y_pred