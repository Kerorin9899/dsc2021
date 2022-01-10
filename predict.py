from scipy import interpolate
import numpy as np
import argparse
import csv
import pdb
from tqdm import tqdm

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchsummary import summary

from Loader import ValidationMydatasets

#from Model import RMInBet
from Model import RMInBet

seed = 7

parser = argparse.ArgumentParser()

parser.add_argument('--test_path', type=str, default="input_data/input_data/test/test_easy.csv")
parser.add_argument('-name','--save_path', type=str)
parser.add_argument('-batch','--train_batch_size', type=int, default=1)
parser.add_argument('-e','--epoch', type=int, default=1000)
parser.add_argument('-lr','--learning_rate', type=float, default=0.01)
parser.add_argument('-len','--length', type=int, default=5)
parser.add_argument('-step', "--scheduler_step", type=int, default=100)
parser.add_argument('-weights','--weights_path', type=str)
args = parser.parse_args()

test_path = args.test_path
train_batch_size = args.train_batch_size
epochs = args.epoch
lr = args.learning_rate
length = args.length
save_name = args.save_path
sche_step= args.scheduler_step
save_weights_path = args.weights_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans = torchvision.transforms.Compose([
            transforms.ToTensor()
        ])

mse = nn.MSELoss()
mse.to(device)

def criterion(data, target):

    loss = mse(data, target)

    return loss

###

testset = ValidationMydatasets(trans, path=test_path)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1)

Net = RMInBet(length, device, batch=train_batch_size)
Net.load_state_dict(torch.load(save_weights_path))

optimizer = optim.Adam(Net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

Net.to(device)

#summary(Net, (1, 63), (1, 63))

###

def main():
    print("Program Start!")
    
    Net.eval()

    with open("results/" + save_name + ".csv", 'w', newline="") as file:
        
        writer = csv.writer(file, lineterminator='\n')

        with open(test_path, 'r') as f:
            reader = csv.reader(f)
            r = next(reader)
            writer.writerow(r)

        with tqdm(total=len(testloader), unit="batch") as pbar:
            
            #torch.autograd.set_detect_anomaly(True)
            for batch_idx, (data, key, offset, y_t, mid, ff) in enumerate(testloader): 
                #pdb.set_trace()
                if ff[0][1] == 1:
                    _f = ff.to('cpu').detach().numpy().copy()
                    _f = _f.squeeze(0)
                    _f = _f.tolist()
                    writer.writerow(_f)
                    
                input_tensor = data.to(device)
                input_key = key.to(device)
                offset_tensor = offset.to(device)

                input_tensor = input_tensor.to(torch.float32)
                input_key = input_key.to(torch.float32)
                offset_tensor = offset_tensor.to(torch.float32)

                y_t = y_t.to('cpu').detach().numpy().copy()
                y_t = y_t.squeeze(0)
                y_t = y_t.squeeze(0)

                key = key.to('cpu').detach().numpy().copy()
                key = key.squeeze(0)
                key = key.squeeze(0)

                output = input_tensor
                cell = torch.zeros([len(data), 1, 512])
                cell = cell.to(device)

                #pdb.set_trace()
                mid = mid.squeeze(0)
                ef = mid[:2]

                mid = mid.to('cpu').detach().numpy().copy()
                
                hidden0 = Net.hidden(input_tensor)

                #pdb.set_trace()     
                estimate = np.zeros([length, 63])
                with torch.no_grad():
                    for t in range(1, length+1):
                        output, hidden0, cell = Net(output, input_key, offset_tensor, length-t, hidden0, cell)

                        out = output.to('cpu').detach().numpy().copy()
                        out = out.squeeze(0)
                        out = out.squeeze(0)

                        relative_pos = np.zeros(63)
                        for k in np.arange(3, 63, 3):
                            relative_pos[k] = out[0]
                            relative_pos[k+1] = out[1]
                            relative_pos[k+2] = out[2] 

                        out += relative_pos
                        offset = y_t - out

                        estimate[t-1] = out

                        offset_tensor = torch.from_numpy(offset.astype(np.float32)).clone()
                        offset_tensor = offset_tensor.to(device)
                        offset_tensor = offset_tensor.to(torch.float32)
                        offset_tensor = offset_tensor.unsqueeze(0)
                        offset_tensor = offset_tensor.unsqueeze(0)

                #pdb.set_trace()
                e = y_t - estimate[length-1]
                #pdb.set_trace()
                for t in range(1, length):
                    ids = [int(f) for f in ef]
                    omega = 1 - ((length - t) / length)

                    es = estimate[t-1]
                    es += omega * e
                    es = es.tolist()

                    ids[1] -= (length - t)
                    ids.extend(es)
                    writer.writerow(ids)
                
                end = mid.tolist()
                writer.writerow(end)
                    
                pbar.update(1)

    print("Finished!")


if __name__ == "__main__":
    main()
