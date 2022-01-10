import sys
import csv
import pdb

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset

class Mydatasets(Dataset):
    def __init__(self, transform=None, path=None, length=5, full=False):
        self.trans = transform

        i = 0
        self.data = []
        self.l = length + 1

        self.input_d = []
        self.input_k = []
        self.input_yt = []
        self.input_offset = []
        self.input_l = []
        self.input_length = []

        with open(path) as f:
            reader = csv.reader(f)
            next(reader)

            if full == True:
                for row in reader:
                    int_row = [float(d) for d in row]
                    self.data.append(int_row)
                    
                    i += 1
            else:
                for row in reader:
                    if int(row[0]) % 4 != 0:
                        int_row = [float(d) for d in row]
                        self.data.append(int_row)
                        
                        i += 1

        num = 0
        root_previous = np.zeros(3)
        x_sum = np.zeros(63)
  
        for idx, r in enumerate(self.data):
            #print(str(idx+self.l) + "/" + str(i))
            if idx + self.l >= i:
                print(idx+self.l)
                break

            frame_id = int(r[0])
            frame_num = int(r[1])
            root = np.array(r[2:5])
            
            #root_previous = root

            if int(r[0]) != int(self.data[idx + self.l][0]):
                continue
            else:

                root = np.array(r[2:5])
                #vector = root - root_previous
                vector = root
                #root_previous = root
                j_t = r[5:]

                relative_pos = np.zeros(60)
                for k in np.arange(0, 60, 3):
                    relative_pos[k] = root[0]
                    relative_pos[k+1] = root[1]
                    relative_pos[k+2] = root[2]

                j_t -= relative_pos
                x_t = np.concatenate([vector, j_t])

                key_frame = self.data[idx + self.l][2:]
                key_frame = np.array(key_frame)

                vector = key_frame[:3]

                relative_pos = np.zeros(63)
                for k in np.arange(3, 63, 3):
                    relative_pos[k] = vector[0]
                    relative_pos[k+1] = vector[1]
                    relative_pos[k+2] = vector[2]

                y_t = np.copy(key_frame)
                key_frame -= relative_pos
                key_frame = key_frame[np.newaxis,:]

                out_data = x_t
                out_data = out_data[np.newaxis,:]

                out_label = np.zeros([self.l - 2, 63])
                out_offset = np.zeros([self.l - 2, 63])
                for j in range(0, self.l - 2):
                    in_data = np.array(self.data[idx+j][2:])

                    offset = np.array(self.data[idx+j][2:])
                    offset = y_t - offset

                    relative_pos = np.zeros(63)
                    for k in np.arange(3, 63, 3):
                        relative_pos[k] = in_data[0]
                        relative_pos[k+1] = in_data[1]
                        relative_pos[k+2] = in_data[2]

                    out_label[j] = in_data - relative_pos

                    relative_pos = np.zeros(63)
                    for k in np.arange(3, 63, 3):
                        relative_pos[k] = offset[0]
                        relative_pos[k+1] = offset[1]
                        relative_pos[k+2] = offset[2]

                    offset -= relative_pos
                    out_offset[j] = offset

                #x_sum += x_t
                y_t = y_t[np.newaxis,:]

                self.input_d.append(out_data)
                self.input_k.append(key_frame)
                self.input_l.append(out_label)
                self.input_offset.append(out_offset)
                self.input_yt.append(y_t)

                num += 1
        

        #pdb.set_trace()

        #x_mean = x_sum / num
        #random.shuffle(self.input_d)
        #random.shuffle(self.input_l)

        self.datanum = num

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.input_d[idx]
        out_key = self.input_k[idx]
        out_label = self.input_l[idx]
        out_offset = self.input_offset[idx]
        out_yt = self.input_yt[idx]

        # if self.trans:
        #     out_data = self.trans(out_data)
        #     out_key = self.trans(out_key)
        #     out_label = self.trans(out_label)

        return out_data, out_key, out_offset, out_label, out_yt

class ValidationMydatasets(Dataset):
    def __init__(self, transform=None, path=None):
        self.trans = transform

        i = 0
        self.data = []

        self.ff = []
        self.input = []
        self.input_d = []
        self.input_k = []
        self.offset = []
        self.y_T = []
        self.mid = []

        with open(path) as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if '' in row:
                    continue

                int_row = [float(d) for idx, d in enumerate(row)]

                self.data.append(int_row)

        num = 0
        root_previous = np.zeros(3)
        x_sum = np.zeros(63)

        i = len(self.data)
        for idx, r in enumerate(self.data):
            #print(str(idx+self.l) + "/" + str(i))
            if idx >= i-1:
                break

            frame_id = int(r[0])
            frame_num = int(r[1])
            root = np.array(r[2:5])
            
            #root_previous = root

            if int(r[0]) != int(self.data[idx + 1][0]):
                continue
            else:

                root = np.array(r[2:5])
                #vector = root - root_previous
                vector = root
                #root_previous = root
                j_t = r[5:]

                relative_pos = np.zeros(60)
                for k in np.arange(0, 60, 3):
                    relative_pos[k] = root[0]
                    relative_pos[k+1] = root[1]
                    relative_pos[k+2] = root[2]

                j_t -= relative_pos
                x_t = np.concatenate([vector, j_t])

                key_frame = self.data[idx + 1][2:]
                key_frame = np.array(key_frame)
                y_t = np.copy(key_frame)
                y_t = y_t[np.newaxis,:]

                vector = key_frame[:3]

                relative_pos = np.zeros(63)
                for k in np.arange(3, 63, 3):
                    relative_pos[k] = vector[0]
                    relative_pos[k+1] = vector[1]
                    relative_pos[k+2] = vector[2]



                key_frame -= relative_pos
                key_frame = key_frame[np.newaxis,:]

                mid = np.array(self.data[idx + 1])

                out_data = x_t
                out_data = out_data[np.newaxis,:]

                offset = np.array(self.data[idx + 1][2:])
                offset = offset - np.array(r[2:])
                offset = offset[np.newaxis,:]

                ff = np.array(r)

                #pdb.set_trace()
                self.input_d.append(out_data)
                self.input_k.append(key_frame)
                self.offset.append(offset)
                self.y_T.append(y_t)
                self.mid.append(mid)
                self.ff.append(ff)

                num += 1
        
        #x_mean = x_sum / num
        #random.shuffle(self.input_d)
        #random.shuffle(self.input_l)

        self.datanum = num

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.input_d[idx]
        out_key = self.input_k[idx]
        out_offset = self.offset[idx]
        out_yt = self.y_T[idx]
        frame_id = self.mid[idx]
        ff = self.ff[idx]

        # if self.trans:
        #     out_data = self.trans(out_data)
        #     out_key = self.trans(out_key)
        #     out_label = self.trans(out_label)

        return out_data, out_key, out_offset, out_yt, frame_id, ff
