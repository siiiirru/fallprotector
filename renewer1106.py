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
        self.dx:float=0
        self.dy:float=0
        self.r_H:float=None
        self.previous:"YoloObj"=None
    def getOriginalXY(self):
        return (
            self.x1*MULTI,self.y1*MULTI,
            self.x2*MULTI,self.y2*MULTI
        )
    def check(self) -> bool:
        fX,fY=0,0
        if self.previous is not None:
            #연속된 obj면
            fX=self.dx>self.W*0.05
            fY=self.dy>self.H*0.05
            isActive=fX and fY
            if isActive:
                print(f"persist obj , x1={self.x1}")
                return True
            #연속적이면서 정적인 객체는 YOLO_Q에 넣지않음
            else: 
                print(f"not persist obj , x1={self.x1}")
                return False
        else:
            #연속되지 않은 값이면 그냥 true 반환
            return True
    def isSame(self,other:"YoloObj") -> bool:
        if other is not None:
            fX=abs(self.X-other.X)<self.W*0.3
            fY=abs(self.Y-other.Y)<self.H*0.3
            if fX and fY:
                self.dx=(self.X-other.X)/2
                self.dy=(self.Y-other.Y)/2
                #찾았으면 이전 값을 prev로 설정
                other.previous=None
                self.previous=other                
                return True
            return False
        #이전 프레임 큐가 없을 때
        else:
            self.dx=0
            self.dy=0
            return True
    def calDifferRealH(self,r_H):
        self.r_H=r_H
        if self.previous==None:
            return 1
        else:
            return r_H/self.previous.r_H


class YOLO_OBJ_Q:
    def __init__(self):
        self.YoloQ=Queue(100)
        self.previousYolo=None
    def putYolo(self,boxes:list[YoloObj]):
            Y_Q=self.YoloQ
            print(f"detected: {len(boxes)} persons")###############
            if Y_Q.qsize()!=100:
                #이전 프레임 큐가 있을때
                if self.previousYolo is not None:
                    for obj in boxes:
                        for p_obj in self.previousYolo:
                            if obj.isSame(p_obj):
                                break
                            
                        if obj.check():
                            Y_Q.put(obj)
                #이전 프레임 큐가 없을때
                else:
                    for obj in boxes:
                            obj.isSame(None)
                            obj.check()
                            Y_Q.put(obj)
            else:
                print("Yolo Queue overflow!!")
            print(f"put YoloQ: {Y_Q.qsize()} objs")
    
    #Yolo큐에 있는 것들 미디어파이프,XG부스트에서 쓰게 주면서 현재Yolo큐를 previous로 등록
    def getYolo(self) -> list[YoloObj]:
        Y_Q=self.YoloQ
        li=[None]*Y_Q.qsize()
        i=0
        while Y_Q.qsize()!=0:
            li[i]=Y_Q.get()
            i+=1
        self.previousYolo=li
        return li
    
addr=Path("/home/fallprotector/project/test_video/not_fall/cleaning_room")
iterator = addr.glob("*")

def main():
    global FALL_COUNTER
    print("Started!!!")
    YOLO_Q=YOLO_OBJ_Q()
    while RUNNING:
        try:
            im_addr = next(iterator) 
            image = cv2.imread(im_addr)
        except StopIteration:
            break

        if image is not None:
                resized_image = cv2.resize(image, YOLO_SIZE)
                cv2.imshow("Processed Image",resized_image)

                results=model(resized_image)
                predictions = results.pred[0]
                person_predictions = predictions[predictions[:, -1] == 0]
                frame_obj_list=[None]*len(person_predictions)
                i=0
                for box in person_predictions:
                    x1,y1,x2,y2=map(int,box[:4])
                    obj=YoloObj(x1,y1,x2,y2)
                    frame_obj_list[i]=obj
                    i+=1
                YOLO_Q.putYolo(frame_obj_list)

                yoloList=YOLO_Q.getYolo()
                is_frame_fall=False
                MD_error_ActiveObj=False
                for yoloObj in yoloList:
                    x1,y1,x2,y2=yoloObj.getOriginalXY()
                    croppedImage=image[y1:y2,x1:x2]
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
                        r_H=hRatio*yoloObj.H #코의 실제위치
                        r_ratioH=yoloObj.calDifferRealH(r_H)
                        if r_ratioH<0.7:
                            features[:33]/=features[23]
                            features[33:]/=features[23+33]
                            features=features.reshape(1,-1)
                            data_scaled = scaler.transform(features)
                            dmatrix = xgb.DMatrix(data_scaled)
                            prediction = xgb_model.predict(dmatrix)
                            if prediction[0]>=0.6:
                                is_frame_fall=True
                                print(f"fall predicion occurred: {prediction[0]}")
                    elif yoloObj.previous is not None: MD_error_ActiveObj=True
                if is_frame_fall: 
                    FALL_COUNTER+=1
                    if FALL_COUNTER>=2:
                        print("!!!real fall occurred!!!")
                        # alert()
                        FALL_COUNTER=0
                elif MD_error_ActiveObj!=True : FALL_COUNTER=0

    pose.close()
    picam2.close()
    print("end")

if __name__ == "__main__":
    main()