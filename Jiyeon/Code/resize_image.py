import cv2
import os

#target_size 미지정시 기본적으로 128, 128 사이즈로 조정한다.
def resize_images(input_folder, output_folder, desired_size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)

        # 이미지 크기 조정
        resized = cv2.resize(img, desired_size)

        # 출력 경로에 크기가 조정된 이미지 저장
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, resized)

# 실행
input_folder_path = 'input_folder'
output_folder_path = 'output_folder'
target_size = (512, 512)  # 원하는 크기로 변경 가능
resize_images(input_folder_path, output_folder_path, target_size)
