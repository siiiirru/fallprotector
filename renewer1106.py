import warnings
import cv2
import torch
import mediapipe as mp
import numpy as np
import xgboost as xgb
from picamera2 import Picamera2
import threading
import time
from queue import Queue, Empty
import signal
import joblib
from pathlib import Path

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

YOLO_THREAD_SIZE=2
FRAME_INTERVAL_MS=1000
FRAME_INTERVAL_S=FRAME_INTERVAL_MS/1000
ORIGINAL_SIZE=(1920, 1080)
FALL_COUNTER=0
RUNNING=True
MULTI=8 # Yolo 삽입 이미지와 원본 이미지 해상도 차이
YOLO_SIZE=tuple(map(int,[i/MULTI for i in ORIGINAL_SIZE]))
YOLO_IOU=0.5
YOLO_CONF=0.34

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.iou = YOLO_IOU
model.conf = YOLO_CONF
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,min_detection_confidence=0.5)
# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": ORIGINAL_SIZE}
)
camera_config["controls"]["FrameDurationLimits"] = (430000, 430000) 
picam2.configure(camera_config)
picam2.start()
# Load XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model.json')
# Load scaler
scaler = joblib.load('scaler.pkl')

def signal_handler(sig, frame):
    """Handle termination signal."""
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

class YoloObj:
    def __init__(self,x1:int,y1:int,x2:int,y2:int):
        self.x1:int=x1
        self.y1:int=y1
        self.x2:int=x2
        self.y2:int=y2
        self.X:float=(x1+x2)/2
        self.Y:float=(y1+y2)/2
        self.W:int=x2-x1
        self.H:int=y2-y1
        self.fCounter:int=1
        self.dx:float=None
        self.dy:float=None
        self.r_H:float=None
        self.previous:"YoloObj"=None
    def getOriginalXY(self):
        return (
            self.x1*MULTI,self.y1*MULTI,
            self.x2*MULTI,self.y2*MULTI
        )
    def check(self) -> bool:
        fX=self.dx>self.W*0.05
        fY=self.dy>self.H*0.05
        isActive=fX and fY
        if isActive:
            self.fCounter=0
        else:
            self.fCounter+=1
            if self.fCounter==2:
                return False
        return True
    def isSame(self,other:"YoloObj") -> bool:
        if other is not None:
            fX=abs(self.X-other.X)<self.W*0.3
            fY=abs(self.Y-other.Y)<self.H*0.3
            if fX and fY:
                self.dx=(self.X-other.X)/2
                self.dy=(self.Y-other.Y)/2
                other.previous=None
                self.previous=other                
                return True
            return False
        else:
            self.fCounter=0
            self.dx=0
            self.dy=0
            return True
    def calDifferRealH(self,r_H):
        self.r_H=r_H
        if self.previous==None:
            return 0
        else:
            return r_H/self.previous.r_H
class QObj:
    def __init__(self):
        self.ImageQ=Queue(10)
        self.YoloQ=[Queue(100),Queue(100),Queue(100)]
        self.CurrentImage=[None,None,None]
        self.YoloPreparedForSXT=[False,False,False]
        self.YoloStared=[False,False,False]
        self.previousYolo=None
    def putImage(self,image:np.ndarray) -> None:
        if self.ImageQ.qsize()!=10:
            self.ImageQ.put(image)
        else:
            self.ImageQ.get()
            self.ImageQ.put(image)
            print("Image Queue overflow!!")
    def getImage(self) -> np.ndarray|None:
        if self.ImageQ.qsize()!=0:
            return self.ImageQ.get()
        else:
            return None
    def putYolo(self,t_id:int,boxes:list[YoloObj],image):
        self.YoloStared[t_id]=True
        self.CurrentImage[t_id]=image
        t_Q=self.YoloQ[t_id]
        print(f"detected: {len(boxes)}")###############
        if self.previousYolo is not None:
            for obj in boxes:
                if t_Q.qsize()!=100:
                    if len(self.previousYolo):
                        for p_obj in self.previousYolo:
                            if obj.isSame(p_obj):
                                break
                        if obj.check():
                            t_Q.put(obj)
                    else:
                        t_Q.put(obj)
                else:
                    print("Yolo Queue overflow!!")
        else:
            for obj in boxes:
                if t_Q.qsize()!=100:
                    obj.isSame(None)
                    obj.check()
                    t_Q.put(obj)
                else:
                    print("Yolo Queue overflow!!")
        print(f"passed: {t_Q.qsize()}")
        self.YoloPreparedForSXT[t_id]=True
    def getYolo(self,t_id:int) -> list[YoloObj]:
        self.YoloPreparedForSXT[t_id]=False
        t_Q=self.YoloQ[t_id]
        li=[None]*t_Q.qsize()
        i=0
        while t_Q.qsize()!=0:
            li[i]=t_Q.get()
            i+=1
        self.previousYolo=li
        return li
    def checkPossible(self,t_id:int):
        return self.YoloPreparedForSXT[t_id]
    def getCurrentImage(self,t_id:int):
        return self.CurrentImage[t_id]
    def isYoloStart(self,t_id:int):
        if self.previousYolo is None:
            return t_id==0
        else:
            if self.YoloStared[(t_id+2)%YOLO_THREAD_SIZE] and not self.YoloPreparedForSXT[t_id]:
                self.YoloStared[(t_id+2)%YOLO_THREAD_SIZE]=False
                return True
            else:
                return False
Q=QObj()
addr=Path("/home/fallprotector/project/test_video/not_fall/cleaning_room")
def ImageThread():
    # while RUNNING:
    #     image=picam2.capture_array()
    #     if image is None or image.size == 0:
    #         continue
    #     Q.putImage(image)
    #     time.sleep(FRAME_INTERVAL_MS/1000)
    i=0
    for im_addr in addr.glob("*"):
        image=cv2.imread(im_addr)
        Q.putImage(image)
        time.sleep(FRAME_INTERVAL_S)
        print(i)
        i+=1
    RUNNING=False
def YoloThread(t_id):
    while RUNNING:
        if Q.isYoloStart(t_id):
            image=Q.getImage()
            if image is not None:
                resized_image = cv2.resize(image, YOLO_SIZE)
                results=model(resized_image)
                predictions = results.pred[0]
                person_predictions = predictions[predictions[:, -1] == 0]
                li=[None]*len(person_predictions)
                i=0
                for box in person_predictions:
                    x1,y1,x2,y2=map(int,box[:4])
                    li[i]=YoloObj(x1,y1,x2,y2)
                    i+=1
                Q.putYolo(t_id,li,image)
        else:
            time.sleep(FRAME_INTERVAL_S/10)

def SkleltonXgboostThread():
    t_id=0
    p_id=0
    global FALL_COUNTER
    while RUNNING:
        if p_id!=t_id:
            print(t_id)
            p_id=t_id
        if Q.checkPossible(t_id):
            yoloList=Q.getYolo(t_id)
            image=Q.getCurrentImage(t_id)
            for yoloObj in yoloList:
                x1,y1,x2,y2=yoloObj.getOriginalXY()
                croppedImage=image[x1:x2,y1:y2]
                skeletons=pose.process(croppedImage)
                if skeletons.pose_landmarks:
                    landmarks=skeletons.pose_landmarks.landmark
                    maxY=-999
                    minY=999
                    nose=landmarks[0].y
                    features=np.zeros(66,float)
                    for i in range(33):
                        Y=landmarks[i].y
                        if maxY<Y:
                            maxY=Y
                        if minY>Y:
                            minY=Y
                        features[i]=landmarks[i].x
                        features[i+33]=Y
                    hRatio=(nose-minY)/(maxY-minY)
                    r_H=hRatio*yoloObj.H
                    r_ratioH=yoloObj.calDifferRealH(r_H)
                    if r_ratioH<0.7:
                        features[:33]/=features[23]
                        features[33:]/=features[23+33]
                        features=features.reshape(1,-1)
                        data_scaled = scaler.transform(features)
                        dmatrix = xgb.DMatrix(data_scaled)
                        prediction = xgb_model.predict(dmatrix)
                        if prediction[0]>=0.6:
                            FALL_COUNTER+=1
                            print(f"fall predicion occurred: {prediction[0]}")
            t_id=(t_id+1)%YOLO_THREAD_SIZE
            if FALL_COUNTER>=2:
                print("!!!real fall occurred!!!")
        else:
            time.sleep(FRAME_INTERVAL_S/10)
print("Started!!!")
image_thread=threading.Thread(target=ImageThread)
yolo_threads:list[threading.Thread]=[]
for i in range(YOLO_THREAD_SIZE):
    yolo_threads.append(threading.Thread(target=YoloThread,args=(i,)))
skeleton_xgboost_thread=threading.Thread(target=SkleltonXgboostThread)

image_thread.start()
for yolo_thread in yolo_threads:
    yolo_thread.start()
skeleton_xgboost_thread.start()

image_thread.join()
for yolo_thread in yolo_threads:
    yolo_thread.join()
skeleton_xgboost_thread.join()

pose.close()
picam2.close()
print("end")
