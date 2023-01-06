import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import PositionalEmbedding

class Transformer(nn.Module):

    def __init__(self, 
         encoding_length,
         decoding_length,
         d_model,
         enc_in_features,
         dec_in_features,
         out_features,                 
         n_heads,
         n_layers,
         dim_feedforward,
         dropout
        ):         
        super(Transformer,self).__init__()
        
        self.encoding_length = encoding_length
        self.decoding_length = decoding_length
        
        self.encoder_input_layer = nn.Linear(
            in_features=enc_in_features, 
            out_features=d_model 
            )

        self.decoder_input_layer = nn.Linear(
            in_features=dec_in_features,
            out_features=d_model
            )  
        
        self.linear_mapping = nn.Linear(
            in_features=d_model, 
            out_features=out_features
            )

        self.positional_encoding_layer = PositionalEmbedding(
            d_model=d_model,
            dropout=dropout
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,          
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers, 
            norm=None
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers, 
            norm=None
            )
        
    # def generate_square_subsequent_mask(self, sz): # tgt에만 적용, src에도 적용하려면 sz dim이 달라져야함 
    #     return torch.triu(
    #         torch.full((sz, sz), float('-inf')), diagonal=1
    #     ).to(device)
    
    def generate_square_subsequent_mask(self, dim1, dim2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu") ): # 221212 변경
        return torch.triu(
            torch.full((dim1, dim2), float('-inf')), diagonal=1
        ).to(device)
    

    def forward(self, src, tgt):
        # print('before input_layer src:', src.shape)
        src = self.encoder_input_layer(src)         
        # print('after innputlayer src:',src.shape)
        src = self.positional_encoding_layer(src)         
        src = self.encoder(src=src)        # src쪽에는 mask 미적용        
        # print('after encoder src:', src.shape)        
        
        
        # print('before dec_input_laer tgt:', tgt.shape)
        decoder_output = self.decoder_input_layer(tgt) 
        # print('before input_laer tgt_y:', decoder_output.shape)
        decoder_output = self.positional_encoding_layer(decoder_output) #최원준 교수: 디코더에도 PE적용         
        tgt_mask = self.generate_square_subsequent_mask(dim1=decoder_output.size(1), dim2=decoder_output.size(1))
        src_mask = self.generate_square_subsequent_mask(dim1=decoder_output.size(1), dim2=src.size(1)) # 221212 추가  
        
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask = src_mask
            )        
        # print('after decoder tgt_y:', decoder_output.shape)
        decoder_output = self.linear_mapping(decoder_output) 
        # print('after linedar mapping tgt_y:', decoder_output.shape)

        return decoder_output