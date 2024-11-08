import cv2
import mediapipe as mp
import numpy as np
import torch
import xgboost as xgb
import joblib

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 이미지 경로
addr = "/home/fallprotector/project/test_video/not_fall/cleaning_room/"
addi = 0
RUNNING = True
FALL_COUNTER = 0
YOLO_SIZE = (640, 360)  # YOLO의 이미지 크기 (임시)

# XGBoost 모델 로드 (예시)
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_model.json')

# 스케일러 로드
scaler = joblib.load('scaler.pkl')

def main():
    global FALL_COUNTER
    global RUNNING
    global addi
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    print("Started!!!")
    while RUNNING:
        try:
            im_addr = addr + str(addi) + ".jpg"
            image = cv2.imread(im_addr)
            addi += 10
        except Exception:
            RUNNING = False
            break
        if image is not None:
            # 원본 이미지에서 MediaPipe 처리 (이미지를 RGB로 변환)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            skeletons = pose.process(image_rgb)

            # 배경 차분 적용
            fg_mask = bg_subtractor.apply(image)
            
            # 특정 픽셀 값 이상을 움직임으로 간주 (임계값 설정)
            motion_detected = np.sum(fg_mask) > 50000  # 임계값 조정

            if motion_detected:
                # 움직임이 감지되면, 외곽선 추출
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # 너무 작은 영역은 제외 (예: 잡음)
                        # 외곽선에 대해 경계 박스를 그리기
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 녹색 경계 박스

            
            if skeletons.pose_landmarks:
                landmarks = skeletons.pose_landmarks.landmark
                maxY = -999
                minY = 999
                nose = landmarks[0].y  # 코의 y 좌표
                features = np.zeros(66, float)
                '''
                # 랜드마크를 순회하면서 이미지에 점과 선을 그린다
                for i in range(33):
                    Y = landmarks[i].y
                    if maxY < Y:
                        maxY = Y
                    if minY > Y:
                        minY = Y
                    features[i] = landmarks[i].x
                    features[i + 33] = Y

                    # 랜드마크 점 그리기
                    x = int(landmarks[i].x * image.shape[1])  # x 좌표 (이미지 크기에 맞게 스케일링)
                    y = int(landmarks[i].y * image.shape[0])  # y 좌표 (이미지 크기에 맞게 스케일링)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 초록색 점

                # 랜드마크 간 연결 (신체 부위 연결)
                connections = mp_pose.POSE_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    start_landmark = landmarks[start_idx]
                    end_landmark = landmarks[end_idx]

                    start_x = int(start_landmark.x * image.shape[1])
                    start_y = int(start_landmark.y * image.shape[0])
                    end_x = int(end_landmark.x * image.shape[1])
                    end_y = int(end_landmark.y * image.shape[0])

                    # 랜드마크 간 선 그리기
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

                # 코의 실제 위치로 비율 계산 (hRatio)
                hRatio = (nose - minY) / (maxY - minY) if (maxY - minY) != 0 else 0.9
                print(f"Height Ratio: {hRatio}")
                
                # XGBoost 예측 (선택 사항)
                features = features.reshape(1, -1)
                data_scaled = scaler.transform(features)
                dmatrix = xgb.DMatrix(data_scaled)
                prediction = xgb_model.predict(dmatrix)
                if prediction[0] >= 0.6:
                    print(f"[fall] Prediction occurred: {prediction[0]}")
                else:
                    print(f"[not fall] Prediction occurred: {prediction[0]}")
                '''
        else:
            print("image end")
            RUNNING=False
                

            # 결과 이미지 표시
            cv2.imshow("Processed Image", image)
            cv2.waitKey(1)  # 1ms 대기, 계속해서 화면을 갱신

    pose.close()
    cv2.destroyAllWindows()
    print("End")

if __name__ == "__main__":
    main()
