import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다. 다른 카메라를 사용하려면 번호를 변경하세요.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 모델로 탐지 수행
    results = model(frame)

    # 탐지된 결과를 화면에 표시
    render_img = results.render()[0]
    cv2.imshow('Camera', render_img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
