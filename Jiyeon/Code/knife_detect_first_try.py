import torch
import cv2

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_ilhoon.pt')

# 카메라 초기화
cap = cv2.VideoCapture(0)  # 0번 카메라 사용. 노트북 내장 카메라가 아닌 외부 카메라를 사용할 경우 다른 번호(예: 1)를 사용해야 할 수도 있습니다.

while True:
    ret, frame = cap.read()  # 카메라로부터 현재 프레임을 읽어옵니다.
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

cap.release()  # 카메라 해제
cv2.destroyAllWindows()
