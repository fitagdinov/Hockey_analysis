import random

from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image
from typing import List
import matplotlib.pyplot as plt

# Load a model
root = os.getcwd()
folder_model = os.path.join(root, "models", "yolov8s-pose.pt")
folder_input_vids = "input_videos"
folder_input_imgs = os.path.join("data", "input_images")
folder_temp_imgs = "temp_images"
folder_output_images = "output_folder"

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 2
color = (0, 0, 255)
thickness = 2


def predict(path:str):
    cv_img = cv2.imread(path)
    # cv_img =cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    w, h,  _ = cv_img.shape
    cv_img = cv2.putText(cv_img, "Окончание толчка ногой", (100, 100),font,
                                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(os.path.join(root, "output_folder", "output_img.jpg"), cv_img)

predict(r"C:\Users\iii\Documents\Programm\Hockey_analysis\data\input_images\random_persons_for_presentation\front\ptn\IMG_0081.jpg")
