import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):    
    def __init__(self, d_model, max_len=5000, dropout = 0.05):                
        super(PositionalEmbedding, self).__init__()        
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
            * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Change to batch_first format.
        pe = pe.permute((1, 0, 2))

        self.register_buffer('pe', pe)

    def forward(self, x):        
        x = x + self.pe[:, :x.size(1), :] #[B, E, F]
        return self.dropout(x)