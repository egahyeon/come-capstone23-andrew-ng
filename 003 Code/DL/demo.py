import os
import time
from datetime import datetime, timedelta
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

import requests
import json


os.environ["CUDA_VISIBLE_DEVICES"]= '3'

seq_lenth = 14

model = YOLO("/home/com_2/workspace/yjshin/capstone/results/runs/detect/train2/weights/best.pt")


class CRNN(nn.Module):
    def __init__(self, n_joints, n_hidden, n_categories, n_layer):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
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
    
    
    
def timeseries_dataset(data, row, seq_lenth):
        
    data = pd.concat([data, pd.DataFrame({'Date_time':[row[0]], 'id_0':[row[1]], 'id_1':[row[2]], 'id_2':[row[3]], 'id_3':[row[4]]})], ignore_index=True)

    if len(data) == seq_lenth:
        results = predict(data)
        print('predict :', results)
        graph(data, results)
        data.drop(data.index[0], axis=0, inplace=True)
        
    return data


flag = [False, False, False, False]
def graph(data, results):
    global flag
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot()
    
    for n, i in enumerate(id_n):
        ma[i].append(sum(data[i].astype('int'))/window_size)
        
        if len(ma[i]) == window_size_2:
            time_list[i].append(data['Date_time'].iloc[len(data['Date_time'])-1])
            
            dma[i].append(sum(ma[i])/window_size_2)    
            del(ma[i][0])

            pre = False
            # if results[n] != None and flag == False:
            #     pred_time_list[i].append(results[n])
            #     pred_dma[i].append(dma[i][-1])
            #     pred = True
            #     flag = True
            if results[n] != None and flag[n] == False:
                pre = True
                flag[n] = True
            if results[n] != None:
                pred_time_list[i].append(results[n])
                pred_dma[i].append(dma[i][-1])

            
            # if i == 'id_2':
            #     ax1.plot(time_list[i], dma[i], label='i')
            #     ax1.scatter(pred_time_list[i], pred_dma[i], c='r')
            #     plt.show()
            api(pig_n[i], data['Date_time'].iloc[len(data['Date_time'])-1], sum(ma[i])/window_size_2, pre)
            

def predict(data):
    
    crnn_threshold = 0.3
    pred_threshold = 0.09
    result = [None, None, None, None]

    with torch.no_grad():
        
        for i in id_n:
            tensor_data = torch.tensor(data[i].astype(float).values).float().to(device)
            tensor_data = tensor_data.unsqueeze(dim=0)
            tensor_data = tensor_data.unsqueeze(dim=2)
            output = crnn[i](tensor_data)
            output = torch.tensor([1 if x > crnn_threshold else 0 for x in output]).squeeze()
            pred[i].append(output)
    if len(pred['id_0']) == window_size:
                
        for n, i in enumerate(id_n):
            pred_ma[i].append(sum(pred[i])/window_size)
        
            if len(pred_ma[i]) == 2:
                if pred_ma[i][0] < pred_threshold < pred_ma[i][1]:
                    result[n] = data['Date_time'].iloc[len(data['Date_time'])-1]
                del(pred_ma[i][0])
    
            del(pred[i][0])
        

    return result


datas = []
curr = datetime.now()
def api(pNo, now, act, pre):
    
    data = {
        'pNo' : pNo,
        'now' : now,
        'act' : act,
        'pred' : pre
    }
    
    datas.append({
        'pNo' : pNo,
        'now' : now,
        'act' : act,
        'pred' : pre
    })
    
    url = "http://18.116.46.26/django/pig_info/gpu_server/pig_data/"
    
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=data, headers=headers, verify=False)
    print(f'status_code : {response.status_code}')
    print(f'text : {response.text}')
    with open(f'/home/com_2/workspace/yjshin/capstone/api_json/{curr}.json','w') as f:
        json.dump(datas, f, indent=4)
        
        

with open('/home/com_2/workspace/yjshin/capstone/stall/cam_stall.pkl', 'rb') as f:
    cam_stall = pickle.load(f)

cam_n = 1

if cam_n == 1:
    cam = cam_stall[0]
    streaming_num = '101'
    num = [20, 19, 18, 17]
elif cam_n == 2:
    cam = cam_stall[1]
    streaming_num = '201'
    num = [16, 15, 14, 13]
elif cam_n == 3:
    cam = cam_stall[2]
    streaming_num = '301'
    num = [12, 11, 10, 9]
elif cam_n == 4:
    cam = cam_stall[3]
    streaming_num = '401'
    num = [8, 7, 6, 5]
elif cam_n == 5:
    cam = cam_stall[4]
    streaming_num = '501'
    num = [4, 3, 2, 1]
else:
    pass

window_size = 10
window_size_2 = 5

id_n = ['id_0', 'id_1', 'id_2', 'id_3']

time_list = {}
ma = {}
dma = {}
pred = {}
pred_ma = {}

pred_time_list = {}
pred_dma = {}

pig_n = {}

crnn = {}

dir = '/home/com_2/workspace/yjshin/capstone/result_lstm/1013_234832_crnn_ex/100.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

n_hidden = 64
n_joints = 1
n_categories = 1
n_layer = 3

for n, i in enumerate(id_n):
    time_list[i] = []
    ma[i] = []
    dma[i] = []
    pred[i] = []
    pred_ma[i] = []
    pred_dma[i] = []
    pred_time_list[i] = []
    
    pig_n[i] = num[n]
    
    crnn[i] = CRNN(n_joints, n_hidden, n_categories, n_layer)
    crnn[i].load_state_dict(torch.load(dir, map_location='cuda'))
    crnn[i].to(device)
    crnn[i].eval()


rows = []
pig_table = pd.DataFrame({'Date_time':[], 'id_0':[], 'id_1':[], 'id_2':[], 'id_3':[]})

source = f'rtsp://admin:Gfarm88555080!@183.99.163.146:554/Streaming/Channels/{streaming_num}'

def ear_detect_1m(rows):
    a = np.array(rows)
    a = a.sum(axis=0)
    
    ear_1m = np.where(a>1, '1', '0')
    # ear_1m = np.where(a>5, '1', '0')

    return ear_1m


results = model.predict(source, stream=True, verbose=False)
# results = model.predict('/home/com_2/workspace/pig_video/newpig.mp4', stream=True, verbose=False)
current = datetime.now()
for r in results:

    row = [0, 0, 0, 0]
    boxes = r.boxes
    for box in boxes:
        b = box.xywh[0].tolist()
        c = box.cls
        if b[0] > cam[0] and b[0] < cam[1]:
            row[0] = 1 if str(int(c)) == '0' else 0
        elif b[0] > cam[1] and b[0] < cam[2]:
            row[1] = 1 if str(int(c)) == '0' else 0
        elif b[0] > cam[2] and b[0] < cam[3]:
            row[2] = 1 if str(int(c)) == '0' else 0
        elif b[0] > cam[3] and b[0] < cam[4]:
            row[3] = 1 if str(int(c)) == '0' else 0
        else:
            pass
        
    rows.append(row)
    # if (current + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S') == datetime.now().strftime('%Y-%m-%d %H:%M:%S'):
    if (current + timedelta(seconds=10)).strftime('%Y-%m-%d %H:%M:%S') == datetime.now().strftime('%Y-%m-%d %H:%M:%S'):
        row_1m = ear_detect_1m(rows)
        row_1m = row_1m.tolist()
        row_1m.insert(0, (current + timedelta(seconds=10)).strftime('%Y-%m-%d %H:%M:%S'))

        # row_1m.insert(0, (current + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:00'))

        pig_table = timeseries_dataset(pig_table, row_1m, seq_lenth)

        rows = []
        current = current + timedelta(seconds=10)

        # current = current + timedelta(minutes=1)