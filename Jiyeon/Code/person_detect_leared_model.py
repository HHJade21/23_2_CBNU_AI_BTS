import torch
import cv2
import numpy as np

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='efficientnet_b3.pt')

# 이미지 파일 로드
img_path = "input1.png"  # 이미지 파일 경로를 입력하세요.
img = cv2.imread(img_path) 

# YOLOv5 모델로 탐지 수행
results = model(img)

# 탐지된 결과를 화면에 표시
render_img = results.render()[0]
cv2.imshow('Image', render_img)

# 'q' 키를 누르면 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
