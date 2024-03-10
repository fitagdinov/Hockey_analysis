import random

from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Load a model
root = os.getcwd()
folder_model = os.path.join(root, "models", "yolov8s-pose.pt")
folder_input_vids = "input_videos"
folder_input_imgs = os.path.join("data", "input_images")
folder_temp_imgs = "temp_images"
folder_output_images = "output_folder"


def getFrames(input_name:str):
    """Получает кадры из видео"""

    person_name, side = (input_name.split(".")[0]).split("_")
    if not os.path.exists(os.path.join(folder_input_imgs, person_name)):
        os.makedirs(os.path.join(folder_input_imgs, person_name), exist_ok=True)
    if not os.path.exists(os.path.join(folder_input_imgs, person_name, side)):
        os.makedirs(os.path.join(folder_input_imgs, person_name, side), exist_ok=True)

    video = cv2.VideoCapture(os.path.join(folder_input_vids, input_name))
    ok, frame = video.read()
    count = 0
    while ok:
        if count%10==0:
            cv2.imwrite(os.path.join(folder_input_imgs, person_name, side, "frame%d.jpg".format(count)), frame)
            print('WRITTEN FRAME:',count)
            count+=1
            ok, frame = video.read()
    video.release()

def create_summary_image_from_ds(dataset_folder:str, save_name:str):
    """Создаёт суммарную матрицу (в виде изображения), которая хранит
    все положения ноги"""

    phase = os.path.basename(dataset_folder)
    pose = os.path.basename(os.path.dirname(dataset_folder))
    person = os.path.basename(os.path.dirname(os.path.dirname(dataset_folder)))

    if not os.path.exists(os.path.join(folder_output_images, person)):
        os.makedirs(os.path.join(folder_output_images, person), exist_ok=True)

    if not os.path.exists(os.path.join(folder_output_images, person, pose)):
        os.makedirs(os.path.join(folder_output_images, person, pose), exist_ok=True)

    output_folder = os.path.join(folder_output_images, person, pose, phase)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(os.path.join(output_folder, save_name)):

        w, h, _ = cv2.imread(os.path.join(dataset_folder, random.choice(os.listdir(dataset_folder)))).shape
        white_img = np.zeros((400, 200), dtype=np.uint8)
        cv2.imwrite(os.path.join(output_folder, save_name), white_img)

    model = YOLO(folder_model)  # load an official model

    # first_img = np.zeros((400, 200), dtype=np.uint8)
    # os.remove(os.path.join("output_folder", "SI", "front", "summary.jpg"))
    # cv2.imwrite(os.path.join("output_folder", "SI", "front", "summary.jpg"), first_img)

    #передаём в модель изображение для предикта точек
    for file in os.listdir(dataset_folder)[:150]:
        results = model(source=os.path.join(dataset_folder, file))



if __name__ == '__main__':

    ################################Получаем кадры#################################
    # getFrames("SI_front.mp4")
    #################################Предобработка####################################
    # cur_folder = os.path.join(folder_input_imgs, "SI", "front")
    # for file in os.listdir(cur_folder):
    #     img = cv2.imread(os.path.join(cur_folder, file))
    #     w = 640
    #     x = img.shape[1]//2
    #     crop_img = img[:, x-w//2:x + w//2]
    #     cv2.imwrite(os.path.join(cur_folder, file), crop_img)
        # plt.imshow(crop_img)
        # plt.show()

    ####################################Делаем предобработку##################################


    person = "first_person"
    pose = "pose_fron_left"
    phase = "otn"

    create_summary_image_from_ds(dataset_folder=os.path.join(root, folder_input_imgs, person, pose, phase),
                                 save_name="summary.jpg")


