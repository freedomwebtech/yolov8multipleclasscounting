import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone


model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('vid7.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
cy1=318
cy2=329

tracker=Tracker()




cardown={}
counter1=[]
carup={}
counter2=[]
offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    car=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            car.append(c)
            list.append([x1,y1,x2,y2])

    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        for i in car:
            x3,y3,x4,y4,id=bbox
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            if cy1<(cy+offset) and cy1>(cy-offset):
               cardown[id]=(cx,cy)
            if cy2<(cy+offset) and cy2>(cy-offset):
               carup[id]=(cx,cy)   
            if id in cardown:   
               if cy2<(cy+offset) and cy2>(cy-offset):
                 cv2.circle(frame,(cx,cy),4,(255,255,0),-1)
                 cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                 if counter1.count(id)==0:   
                    counter1.append(id)
            if id in carup:
               if cy1<(cy+offset) and cy1>(cy-offset):
                 cv2.circle(frame,(cx,cy),4,(255,255,0),-1)
                 cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                 if counter2.count(id)==0:   
                    counter2.append(id)
                
    
             
    cardownc=(len(counter1))
    carupc=(len(counter2))

 

    cv2.line(frame,(150,cy1),(896,cy1),(255,0,255),2)
    cv2.line(frame,(138,cy2),(916,cy2),(0,255,0),2)
    cvzone.putTextRect(frame,f'carupc:-{carupc}',(885,71),1,1)
    cvzone.putTextRect(frame,f'cardownc:-{cardownc}',(24,66),1,1)





    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

