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
import math
import joblib
from pathlib import Path
from sendEmail import sendMsg

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message=".*GetPrototype() is deprecated.*")
warnings.filterwarnings('ignore', message=".*X does not have valid feature names.*")


FRAME_INTERVAL_MS=1000
FRAME_INTERVAL_S=FRAME_INTERVAL_MS/1000
ORIGINAL_SIZE=(1920, 1080)
VIEW_SIZE=(1920//2,1080//2)
FALL_COUNTER=0
WINDOW_NAME = "Processed Image"
RUNNING=True
MULTI=1 # Yolo 삽입 이미지와 원본 이미지 해상도 차이
MP_SIZE=tuple(map(int,[i/MULTI for i in ORIGINAL_SIZE]))
QSIZE=5
NOSE_FALL_RATIO=0.20
DISTANCE_RATIO=2.35

folder_name = {
    #not_fall
    0: 'gymnastics',
    1: 'organizing_item',
    2: 'cleaning_room',
    3: 'rest',
    #fall
    4: 'common_fall',
    5: 'picture_fall'
}

addi = 0
addr="/home/fallprotector/project/test_video/fall/"+folder_name[4]+"/"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2,min_detection_confidence=0.7,min_tracking_confidence=0.5)
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
xgb_model.load_model('xgboost_model1111.json')
# Load scaler
scaler = joblib.load('scaler1111.pkl')

def signal_handler(sig, frame):
    """Handle termination signal."""
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

class MpObj:
    def __init__(self,minX:int,minY:int,maxX:int,maxY:int,nose:int):
        self.minX:int=minX
        self.minY:int=minY
        self.maxX:int=maxX
        self.maxY:int=maxY
        self.X:float=(minX+maxX)/2
        self.Y:float=(minY+maxY)/2
        self.W:int=maxX-minX
        self.H:int=maxY-minY
        self.nose:int=nose
        self.r_H:float=None
    def getOriginalXY(self):
        return (
            self.minX*MULTI,self.minY*MULTI,
            self.maxX*MULTI,self.maxY*MULTI
        )
def calDifferRealH(current_r_H,prev_r_H):
    if prev_r_H==None:
        return 1
    return current_r_H/prev_r_H

def drawCircle(landmark,croped_image_width,croped_image_height,resized_image):
    x = int(landmark.x * croped_image_width)  # x 좌표 (이미지 크기에 맞게 스케일링)
    y = int(landmark.y * croped_image_height)  # y 좌표 (이미지 크기에 맞게 스케일링)
    cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)  # 초록색 점
    return x,y

def drawLine(landmarks,croped_image_width,croped_image_height,resized_image):
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]
        start_x = int(start_landmark.x * croped_image_width)
        start_y = int(start_landmark.y * croped_image_height)
        end_x = int(end_landmark.x * croped_image_width)
        end_y = int(end_landmark.y * croped_image_height)
        cv2.line(resized_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

def get_floor_distance(landmarks, resize_h):
    # 발목 좌표 (왼쪽 발목과 오른쪽 발목)
    left_ankle_y = landmarks[24].y * resize_h  # 왼쪽 발목
    right_ankle_y = landmarks[28].y * resize_h  # 오른쪽 발목

    # 발목들의 평균을 사용하여 바닥의 위치 추정
    floor_y = (left_ankle_y + right_ankle_y) / 2
    return floor_y

def adjust_nose_fall_ratio(floor_y, nose_y, resize_h):
    # 바닥과 코의 y좌표 차이 계산
    distance_from_floor = floor_y - nose_y
    distance_ratio = (distance_from_floor / resize_h)*DISTANCE_RATIO  # 상대적인 거리 비율 계산

    # 사람과 카메라 사이의 거리가 가까우면 비율을 더 크게 조정
    adjusted_nose_fall_ratio = NOSE_FALL_RATIO * (1 + distance_ratio)
    return adjusted_nose_fall_ratio

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)  # 창 이름 설정
cv2.moveWindow(WINDOW_NAME, 100, 100)

def main():
    global FALL_COUNTER
    global RUNNING
    global addi
    print("Started!!!")
    mpQ=Queue(QSIZE) #put,get으로 사용
    start_time = time.time()
    frame_count = 0
    mp_not_making_counter=0
    while RUNNING:
        try:
            im_addr = addr+str(addi)+".jpg"
            image = cv2.imread(im_addr)
            addi+=8
        except Exception:
            print("read error")
            RUNNING=False
            break

        if image is not None:
            
            resized_image = cv2.resize(image, MP_SIZE)
            skeletons=pose.process(resized_image)
            #resized_image=np.ones(shape=MP_SIZE)
            if skeletons.pose_landmarks:
                mp_not_making_counter=0
                resize_h=resized_image.shape[0]
                resize_w=resized_image.shape[1]
                prevObj=None
                if mpQ.qsize()>0:
                    prevObj=mpQ.get()
                landmarks=skeletons.pose_landmarks.landmark
                x_coords = [landmark.x for landmark in landmarks]
                y_coords = [landmark.y for landmark in landmarks]
                min_x = int(min(x_coords) * resize_w)
                max_x = int(max(x_coords) * resize_w)
                min_y = int(min(y_coords) * resize_h)
                max_y = int(max(y_coords) * resize_h)
                nose=landmarks[0].y*resize_h
                newObj=MpObj(min_x,min_y,max_x,max_y,nose)
                features=np.zeros(66,float)
    
                for i in range(33):
                    Y=(landmarks[i].y*resize_h)-min_y
                    features[i]=(landmarks[i].x*resize_w)-min_x
                    features[i+33]=Y
                    drawCircle(landmarks[i],resize_w,resize_h,resized_image)
                drawLine(landmarks,resize_w,resize_h,resized_image)

                mpQ.put(newObj)

                differNose = nose-prevObj.nose if prevObj!=None else 0 
                if differNose>0:
                    # 발의 위치를 활용하여 바닥의 위치 추정
                    floor_y = get_floor_distance(landmarks, resize_h)
                    # 바닥 위치와 코의 위치를 비교하여 동적으로 NOSE_FALL_RATIO 조정
                    adjusted_nose_fall_ratio = adjust_nose_fall_ratio(floor_y, nose, resize_h)
                    #if differNose>200:
                        #print(resize_h*adjusted_nose_fall_ratio,"dN=",differNose,"addi=",addi)
                    if differNose>(resize_h*adjusted_nose_fall_ratio):
                        features=features.reshape(1,-1)
                        data_scaled = scaler.transform(features)
                        dmatrix = xgb.DMatrix(data_scaled)
                        prediction = xgb_model.predict(dmatrix)
                        if prediction[0]>=0.6 :
                            FALL_COUNTER+=1
                        else :
                            FALL_COUNTER=0
            else :
                mp_not_making_counter+=1
                #print("Failed to generate coordinates in MediaPipe.")
                if mp_not_making_counter>4: 
                    FALL_COUNTER=0
            
            
        
        
            show_image=cv2.resize(resized_image,VIEW_SIZE)
            cv2.imshow(WINDOW_NAME,show_image)
            cv2.waitKey(1)
            if FALL_COUNTER>=1:
                print("[real fall] occurred!!!")
                # alert()
                
                #sendMsg("The Fall has been occurred!",
                        #time.strftime('%Y-%m-%d %H:%M:%S'),show_image,"qw803qw803@gmail.com")
                FALL_COUNTER=0
            '''
            # 매 초마다 FPS 계산
            frame_count += 1
            elapsed_time = time.time() - start_time
            if int(elapsed_time) > 0:
                fps = frame_count / elapsed_time
                print(f"Frames processed per second: {fps:.2f}")
                start_time = time.time()  # Reset time for the next second
                frame_count = 0  # Reset frame count for the next second
                '''
        else: RUNNING=False    
    pose.close()
    picam2.close()
    #cv2.destroyAllWindows()
    print("end")

if __name__ == "__main__":
    main()





