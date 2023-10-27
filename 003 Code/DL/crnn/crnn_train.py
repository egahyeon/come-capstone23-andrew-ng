import torch
import torch.nn as nn
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import random
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from tqdm import tqdm

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(device)


def sliding_windows(data, seq_length):
    x = []
    y = []
    n = data.index[0]

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]

        # 학습 데이터셋 다음을 label로 (len(_x) + 1)
        _y = data.loc[n+seq_length]['label']
        n += 1

        # # 학습 데이터셋 마지막을 label로 (len(_x))
        # _y = data.loc[n+seq_length -1]['label']
        # n += 1

        x.append(_x.drop(columns=['label']))
        y.append([_y])
    

    return np.array(x),np.array(y)


class CustomDataset(Dataset): 
  def __init__(self, data_x, data_y):
    self.data_x = data_x
    self.data_y = data_y

  def __len__(self):
    return self.data_y.shape[0]

  def __getitem__(self, idx):
    x = torch.from_numpy(self.data_x[idx])
    y = torch.from_numpy(self.data_y[idx])
    return x,y
  
  def get_labels(self):
    return list(train_y_.squeeze())


seq_length = 2880

path = '/home/com_2/workspace/pig_dataset/train/*'
file_list = glob.glob(path)
train_files = [file for file in file_list if file.endswith('.csv')]


for i, file in enumerate(train_files):
    pig_data = pd.read_csv(file)
    pig_data = pig_data.drop(columns=['Date_time'])
    if i == 0:
        x_train, y_train = sliding_windows(pig_data, seq_length)
    else:
        _x_train, _y_train = sliding_windows(pig_data, seq_length)
        x_train = np.concatenate((x_train, _x_train),axis=0)
        y_train = np.concatenate((y_train, _y_train),axis=0)



n_hidden = 64
n_joints = 1
n_categories = 1
n_layer = 3
batch_size = 512
num_epochs = 150
learning_rate = 0.0001

train_dataset = CustomDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size = batch_size,
                                          shuffle=True)


                                       
class CRNN(nn.Module):
    def __init__(self, n_joints, n_hidden, n_categories, n_layer):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1440, stride=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1440, stride=1)
        self.relu2 = nn.ReLU()
        
        # LSTM layer
        self.lstm = nn.LSTM(64, n_hidden, num_layers=n_layer, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(64, n_hidden, num_layers=n_layer, batch_first=True)
        
        # FC layer
        self.fc = nn.Linear(n_hidden*2, n_categories)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 입력 데이터의 차원 변경
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = x.permute(0, 2, 1)  # LSTM을 위해 차원 순서 변경
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 마지막 시간 스텝의 결과만 사용
        
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


date = datetime.now().strftime("%m%d_%H%M%S")
folder_name = f'{date}_crnn_gt_1440_1440'
os.mkdir(f"/home/com_2/workspace/yjshin/capstone/result_lstm/{folder_name}")

model = CRNN(n_joints, n_hidden, n_categories, n_layer)
model.to(device)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

all_losses = []
current_loss = 0
_loss = 100

total_step = len(train_loader)
for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(train_loader), total=total_step, leave=False, desc=f'Epoch {epoch+1}/{num_epochs}')


    for i, data in progress_bar:
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        # labels = torch.squeeze(labels)
        
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    if (epoch+1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        all_losses.append(current_loss / total_step)
        current_loss = 0
        # if _loss > loss:
        #     _loss = loss
        torch.save(model.state_dict(), f'/home/com_2/workspace/yjshin/capstone/result_lstm/{folder_name}/{epoch+1}.pt')



torch.save(model.state_dict(), f'/home/com_2/workspace/yjshin/capstone/result_lstm/{folder_name}/last.pt')