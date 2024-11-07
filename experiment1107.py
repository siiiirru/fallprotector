from threading import Thread
from queue import Queue

class QObj:
    def __init__(self):
        self.YoloPreparedForSXT=[False,False,False]
        self.YoloStared=[False,False,False]
        self.previousYolo=None
    def putYolo(self,t_id:int):
        self.YoloStared[t_id]=True
        self.YoloPreparedForSXT[t_id]=True
    def getYolo(self,t_id:int):
        self.YoloPreparedForSXT[t_id]=False
        self.previousYolo=1
    def checkPossible(self,t_id:int):
        return self.YoloPreparedForSXT[t_id]
    def isYoloStart(self,t_id:int):
        if self.previousYolo==None:
            return t_id==0
        else:
            Flag=self.YoloStared[(t_id+2)%3]
            self.YoloStared[(t_id+2)%3]=False
            return Flag
Q=QObj()
def YoloThread(t_id):
    while True:
        if Q.isYoloStart(t_id):
            Q.putYolo(t_id)
def SkleltonXgboostThread():
    t_id=0
    while True:
        print(Q.YoloStared)
        if Q.checkPossible(t_id):
            yoloList=Q.getYolo(t_id)
            t_id=(t_id+1)%3

yolo_threads=[]
for i in range(3):
    yolo_threads.append(Thread(target=YoloThread,args=(i,)))
skeleton_xgboost_thread=Thread(target=SkleltonXgboostThread)
for yolo_thread in yolo_threads:
    yolo_thread.start()
skeleton_xgboost_thread.start()