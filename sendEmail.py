import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from PIL import Image
import io
import cv2
import numpy as np

def sendMsg(subject,body,image_array,receiver_email):
    sender_email = "EMAIL"
    password = "PWD"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body+" fall occurred!!", "plain"))
    
    #cv2.imshow("hello",image_array)
    #cv2.waitKey(0)
    
    # 데이터 타입과 값 범위를 변환합니다.
    if image_array.dtype != np.uint8:
        # 값 범위가 0~1 사이인 경우 0~255로 스케일링
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        print(f"Converted image array dtype: {image_array.dtype}")

    # 이미지 배열이 2차원인 경우 3채널로 변환합니다.
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
        print(f"Expanded image array shape: {image_array.shape}")
    image = Image.fromarray(image_array)
    #if image.mode != "RGB":
    #    image = image.convert("RGB")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    image_attachment = MIMEImage(image_buffer.read(), name=body+"_fall_occurred.jpg")
    msg.attach(image_attachment)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  
        server.login(sender_email, password) 
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.quit()