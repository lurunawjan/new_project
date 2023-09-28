from ultralytics import YOLO
import cv2
from datetime import datetime
import csv
import numpy as np
import pandas as pd
import torch

# import best weights or pre-trained model
model = YOLO("D:/pothole_YOLOv8/pothole_dataset_v8/pothole_dataset_v8/runs/detect/yolov8m_v8_50e12/weights/best.pt")

if torch.cuda.is_available():
    model.to('cpu')
else:
    print("No GPUs available. Using CPU.")

def detect(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x,y]
        print(colorsBGR)

cv2.namedWindow('detect')
cv2.setMouseCallback('detect',detect)

# read video
cap = cv2.VideoCapture('D:/pothole_dataset/HG_road/220929_GX010021_2R.mp4')

# Get the video frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose a different codec
output_path = 'output_new_2.mp4'
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

# read classes (pothole classes)
my_file = open("classes.txt","r")
data = my_file.read()
class_list = data.split("\n")
# count=0

# Region of interest 
ROI = [(3,1008),(686,274),(1069,279),(1573,1008)]

# create empty list
list = []

while True:

    ret, frame = cap.read()
    if ret is None:
        break

    results = model(frame)
    a=results[0].boxes.boxes # tensor data type
    try:
        numpy_array = a.numpy()
    except TypeError as e:
        print(e)
    
    tensor_cpu = a.cpu()
    numpy_array = tensor_cpu.numpy()

    px=pd.DataFrame(numpy_array).astype(float)
    # print(px)

    for index,row in px.iterrows():
        # print(row)
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        cl=float(row[4])
        d=int(row[5])
        c=class_list[d]

        list.append([x1,y1,x2,y2,cl,d,c])
        
        # cv2.resize(frame,(1020,650))
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
    
    cv2.polylines(frame,[np.array(ROI,np.int32)], True, (255,255,0),3)
    cv2.namedWindow("detect")
    cv2.imshow("detect",frame)

    out.write(frame)

    csv_file = 'detection_data_vid2.csv'

    with open(csv_file,'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['Xmin','Ymin','Xmax','Ymax','confidence level','class','id_name']
        csv_writer.writerow(header)

        for row in list:
            csv_writer.writerow(row)
        
    if cv2.waitKey(0)&0XFF==27:
        break

cap.release()
# csv_file.close()
cv2.destroyAllWindows()
