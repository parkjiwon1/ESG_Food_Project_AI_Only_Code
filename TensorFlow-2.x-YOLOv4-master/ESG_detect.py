import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from yolov3.utils import detect_image, Load_Yolo_model, Create_Yolo, make_ingredients_list
from yolov3.configs import *
from yolov3.yolov4 import read_class_names

image_path   = "./IMAGES/Test_img1.jpg" # predict 할 이미지

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES) # 모델 불러오기
yolo.load_weights(f"./checkpoints/yolov4_custom_val_loss_Best")  # 모델 weight 불러오기
NUM_CLASS = read_class_names(TRAIN_CLASSES)  # CLASSES NUM, 모델 서버 시작하기 전에 실행

image,pred_classes = detect_image(yolo, image_path, "./IMAGES/Test_img1_pred.jpg",
                     input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0)) # detect 함수

if len(pred_classes) == 0: print("검출 안 됐음. 다시 사진 업로드 해주세요.")

ingredients_list = make_ingredients_list(NUM_CLASS,pred_classes) # 재료 리스트 벡터화

print(ingredients_list)
