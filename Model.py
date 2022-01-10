import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import pdb

class RMInBet(nn.Module):
    def __init__(self, frame_length, device, dropout_p=0.2, joint_num=21, data_length=63, dim_model=256, batch=128):
        super(RMInBet, self).__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.len_data = data_length
        self.dim = dim_model
        self.batch = batch
        self.device = device

        #Encode
        self.fc_state01 = nn.Linear(self.len_data, 512)
        self.fc_state02 = nn.Linear(512, 256)

        self.fc_offset01 = nn.Linear(self.len_data, 512)
        self.fc_offset02 = nn.Linear(512, 256)
 
        self.fc_target01 = nn.Linear(self.len_data, 512)
        self.fc_target02 = nn.Linear(512, 256)

        self.positionalencoder = PositionalEncoding(dim_model=dim_model, length=frame_length, batch=batch, device=device)


        #self.transformer = nn.Transformer(d_model=dim_model, dropout=dropout_p, dim_feedforward=512)

        if frame_length < 5:
            self.lamda = 0
        elif frame_length >= 5 and frame_length < 30:
            self.lamda = (frame_length - 5) / 25
        else:
            self.lamda = 1

        #before
        self.fc01 = nn.Linear(63, 256)
        self.fc02 = nn.Linear(256, 512)

        #Decode
        self.fc_decode01 = nn.Linear(512 ,256)
        self.fc_decode02 = nn.Linear(256, 128)
        self.fc_decode03 = nn.Linear(128, 63)

        #LSTM
        self.lstm = nn.LSTM(768, 512, batch_first=True)

        #Activation
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

    def StateEncoder(self, x, l):
        x = self.fc_state01(x)
        x = self.lrelu(x)

        x = self.fc_state02(x)
        x = self.lrelu(x)

        x = self.positionalencoder(l, x)

        return x

    def OffsetEncoder(self, x, l):
        x = self.fc_offset01(x)
        x = self.lrelu(x)

        x = self.fc_offset02(x)
        x = self.lrelu(x)

        x = self.positionalencoder(l, x)

        return x

    def TragetEncoder(self, x, l):
        x = self.fc_target01(x)
        x = self.lrelu(x)

        x = self.fc_target02(x)
        x = self.lrelu(x)

        x = self.positionalencoder(l, x)

        return x

    def hidden(self, x):
        x = self.fc01(x)
        x = self.fc02(x)

        return x

    def Decoder(self, x):
        x = self.fc_decode01(x)
        x = self.lrelu(x)

        x = self.fc_decode02(x)
        x = self.lrelu(x)

        x = self.fc_decode03(x)

        return x

    def forward(self, x, keyframe, offset, length, hidden0=None, cell=None):
        s_h = self.StateEncoder(x, length)
        o_h = self.OffsetEncoder(offset, length)
        t_h = self.TragetEncoder(keyframe, length)

        hidden0 = hidden0.transpose(1,0)
        z_target = torch.cat([o_h, t_h], dim=2)
        
        #pdb.set_trace()

        z = torch.cat([s_h, z_target], dim=2)
        cell = cell.transpose(1, 0)
        #z_target = self.positionalencoder(length, z_target)

        # if len(s_h) < self.batch:
        #     pad = torch.zeros([1,1,256]).to(self.device)
            
        #     while(len(s_h) < self.batch):
        #         s_h = torch.cat([s_h, pad])
        #         h_b = torch.cat([h_b, pad])
        #         z_target = torch.cat([z_target, pad])


        h_b = hidden0
        h_b = h_b.transpose(1, 0)
        hidden0 = hidden0.transpose(1, 0)

        h_b = h_b.transpose(1, 0)
        z_target = z_target.transpose(1, 0)

        _, (hide, cell) = self.lstm(z, (hidden0, cell))


        #hidden0 = hide
        h = hide.transpose(1, 0)
        h = self.Decoder(h)
        cell = cell.transpose(1, 0)
        output = x + h

        return output, hide, cell


class PositionalEncoding(nn.Module):
    def __init__(self, length, dropout_p=0.2, max_len=63, dim_model=10000, batch=128, device=None):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.dim = dim_model
        self.batch = batch
        self.device = device
        self.max_len = 1


    def forward(self, length, token_embedding):
        
        pos_encode = torch.zeros(self.max_len, self.dim)
        division_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0)) / self.dim)


        # Residual connection + pos encoding
        positions_list = torch.full((1, int(self.dim/2)), length, dtype=torch.float32)
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encode[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encode[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encode = pos_encode.unsqueeze(0).transpose(0, 1)
        pos_encode = pos_encode.to(self.device)

        #pdb.set_trace()
        
        return token_embedding + pos_encode[:token_embedding.size(0), :]

