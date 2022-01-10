from scipy import interpolate
import numpy as np
import argparse
import csv
import pdb
import sys
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

from Loader import Mydatasets

from Model import RMInBet

seed = 7

parser = argparse.ArgumentParser()

parser.add_argument('--train_path', type=str, default="input_data/train/train.csv")
parser.add_argument('--test_path', type=str, default="input_data/test/test_easy.csv")
parser.add_argument('-name','--save_path', type=str)
parser.add_argument('-batch','--train_batch_size', type=int, default=128)
parser.add_argument('-e','--epoch', type=int, default=1000)
parser.add_argument('-lr','--learning_rate', type=float, default=0.01)
parser.add_argument('-len','--length', type=int, default=45)
parser.add_argument('-full','--Full_Train', type=bool, default=False)
parser.add_argument('-load', '--load_weights', type=str, default=None)
args = parser.parse_args()

train_path = args.train_path
test_path = args.test_path
train_batch_size = args.train_batch_size
epochs = args.epoch
lr = args.learning_rate
length = args.length
save_name = args.save_path
load_weight = args.load_weights
Isfull = args.Full_Train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed)
if device:
    torch.cuda.manual_seed(seed)

trans = torchvision.transforms.Compose([
            transforms.ToTensor()
        ])

mse = nn.MSELoss(reduction="sum")
mse.to(device)

def criterion(data, target):

    loss = mse(data, target)

    return loss

###

print("Test.py")

trainset = Mydatasets(trans, train_path, length, Isfull)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size)

Net = RMInBet(length, device, batch=train_batch_size)

if not load_weight is None:
    Net.load_state_dict(torch.load(load_weight))

optimizer = optim.Adam(Net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

Net.to(device)

alpha = 0.5

###


def main():
    print("Program Start!")
    
    Net.train()
    allloss = 0

    losses = []
    test_losses = []
    hidden0 = None

    min_loss = 1e+5

    for ep in range(1, epochs+1):
        train_loss = 0

        print("Epoch[{}/{}]:".format(ep, epochs))
        sys.stdout.flush()

        with tqdm(total=len(trainloader), unit="batch") as pbar:
            pbar.set_description("Epoch [{}/{}]".format(ep, epochs))
            torch.autograd.set_detect_anomaly(True)
            for batch_idx, (data, key, offset, target, y_t) in enumerate(trainloader): 

                input_tensor = data.to(device)
                input_key = key.to(device)
                target_tensor = target.to(device)
                offset_tensor = offset.to(device)
                y_t_tensor = y_t.to(device)

                input_tensor = input_tensor.to(torch.float32)
                input_key = input_key.to(torch.float32)
                target_tensor = target_tensor.to(torch.float32)
                offset_tensor = offset_tensor.to(torch.float32)
                y_t_tensor = y_t_tensor.to(torch.float32)

                optimizer.zero_grad()

                output = input_tensor

                hidden0 = Net.hidden(input_tensor)
                hidden0 = hidden0.transpose(1,0)

                cell = torch.zeros([len(data), 1, 512])
                cell = cell.to(device)

                off = offset_tensor[:,0,:]
                off = off.unsqueeze(1)

                mean_loss = torch.zeros(length-1)
                for t in range(1, length):
                    if t != 1:
                        if random.random() < alpha:
                            output =  target_tensor[:, t-1, :].unsqueeze(1)
                            off = offset_tensor[:, t-1, :].unsqueeze(1)
                        else:
                            relative_pos = torch.zeros([len(data), 1, 63]).to(device)
                            for k in np.arange(3, 63, 3):
                                relative_pos[:,:,k] = output[:,:,0]
                                relative_pos[:,:,k+1] = output[:,:,1]
                                relative_pos[:,:,k+2] = output[:,:,2]

                            off = output + relative_pos
                            off = y_t_tensor - off

                    cell = cell.to(device)
                    hidden0 = hidden0.to(device)
                    output, hidden0, cell = Net(output, input_key, off, length-t, hidden0, cell)

                    tar = target_tensor[:, t-1, :].unsqueeze(1)

                    mean_loss[t-1] = criterion(output, tar)

                loss = torch.mean(mean_loss)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix({"loss":loss.item()/train_batch_size, "lr":scheduler.get_last_lr()})
                pbar.update(1)

            train_loss /= len(trainloader.dataset)
            losses.append(train_loss)
            print('Epoch [{}/{}], Avg loss: {:.8f}'.format(ep,epochs,train_loss))

        if min_loss > train_loss:
            min_loss = train_loss
            torch.save(Net.state_dict(), 'weights/' + str(save_name) + '.pth')

        if ep % 10 == 0:
            np.save("Loss/" + str(save_name) + '.npy', np.array(losses))
        
    print("Train finished!")


if __name__ == "__main__":
    main()
