import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models

# yolo_nas_s 모델 사용
model = models.get('yolo_nas_s', num_classes= 7, checkpoint_path='weights/ckpt_best.pth')

model.predict_webcam()
