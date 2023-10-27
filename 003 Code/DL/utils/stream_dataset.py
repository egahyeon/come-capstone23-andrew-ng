from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import numpy as np
import pandas as pd
import time
import os
import time
from datetime import datetime, timedelta
import glob
import csv
import pickle

with open('/home/com_2/workspace/yjshin/capstone/stall/cam_stall.pkl', 'rb') as f:
    cam_stall = pickle.load(f)

while True:
    current = datetime.now()
    if current.weekday() == 2: # 3으로 수정

        os.environ["CUDA_VISIBLE_DEVICES"]= '2'

        model = YOLO("/home/com_2/workspace/yjshin/capstone/results/runs/detect/train2/weights/best.pt")

        fields = ['Date_time','id_0','id_1','id_2','id_3']
        rows = []

        cam_n = 4

        if cam_n == 1:
            cam = cam_stall[0]
            streaming_num = '101'
            file_name = 'cam1'
        elif cam_n == 2:
            cam = cam_stall[1]
            streaming_num = '201'
            file_name = 'cam2'
        elif cam_n == 3:
            cam = cam_stall[2]
            streaming_num = '301'
            file_name = 'cam3'
        elif cam_n == 4:
            cam = cam_stall[3]
            streaming_num = '401'
            file_name = 'cam4'
        elif cam_n == 5:
            cam = cam_stall[4]
            streaming_num = '501'
            file_name = 'cam5'
        else:
            pass


        source = f'rtsp://admin:Gfarm88555080!@183.99.163.146:554/Streaming/Channels/{streaming_num}'

        def ear_detect_1m(rows):
            a = np.array(rows)
            a = a.sum(axis=0)
            
            ear_1m = np.where(a>1, '1', '0')
            # ear_1m = np.where(a>5, '1', '0')
            return ear_1m

        # results = model.predict('/home/com_2/workspace/pig_video/newpig.mp4', stream=True, verbose=False)
        results = model.predict(source, stream=True, verbose=False)
        current = datetime.now()
        for r in results:
            if current.weekday() == 2:

                csv_name = f'/home/com_2/workspace/yjshin/capstone/timeseries_dataset/{current.strftime("%Y%m%d")}_{file_name}.csv'

                if not os.path.isfile(csv_name):
                    with open(csv_name, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(fields)
                        
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
            if (current + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S') == datetime.now().strftime('%Y-%m-%d %H:%M:%S'):
            # if (current + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S') == datetime.now().strftime('%Y-%m-%d %H:%M:%S'):
                row_1m = ear_detect_1m(rows)
                row_1m = row_1m.tolist()
                row_1m.insert(0, (current + timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S'))
                # row_1m.insert(0, (current + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:00'))
                with open(csv_name, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_1m)
                rows = []
                # current = current + timedelta(minutes=1)
                current = current + timedelta(seconds=5)