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

def create_summary_image_from_ds(dataset_folder:str, save_name:str, **kwargs):
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
    keypoints = []
    images = []
    boxes = []
    for file in os.listdir(dataset_folder)[:1]:
        results = model(source=os.path.join(dataset_folder, file),  save = True)
        keypoints.append(results[0].keypoints.xy)
        boxes.append(results[0].boxes.xywh)
        images.append(results[0].orig_img)


    create_summary_data_from_points(images = images,
                                    boxes = boxes,
                                    points= keypoints,
                                    output_folder=output_folder, summary_img=save_name)



def create_summary_data_from_points(images:List, boxes:List,  points:List, output_folder,  summary_img):
    """Принимает массив изображений и точек, сохраняет результирующее изображение для классификации"""

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 0, 255)
    thickness = 2

    for img_idx in range(len(images)):
        w, h, _ = images[img_idx].shape
        white_img = cv2.imread(os.path.join(output_folder,  summary_img))
        add_image = np.zeros((w, h, 3), dtype=np.uint8)
        box  = boxes[img_idx][0]
        x1 = int(box[0])
        y1 = int(box[1])
        w = int(box[2])
        h = int(box[3])
        x1 -= w // 2
        y1 -= h // 2
        x2 = x1 + w
        y2 = y1 + h
        for idx_pt, xy in enumerate(points[img_idx][0]):
            if idx_pt in range(11, 17): #Точки положения ног
                if xy.tolist() != [0., 0.]: # and idx_pt == 16:  # правая(задняя) нога
                    add_image[int(xy[1])][int(xy[0])] = 255

                    # results[idx].orig_img = cv2.putText(results[idx].orig_img,
                    #                                     str(idx_pt), (int(xy[0])-30, int(xy[1]-20)),font,
                    #                                     fontScale, color, thickness, cv2.LINE_AA)
                    # results[idx].orig_img = cv2.circle(results[idx].orig_img, (int(xy[0]), int(xy[1])),
                    #                                    radius=10, thickness=-1, color=color)
    # plt.imshow(results[idx].orig_img)
    # plt.show()
    # Заглушка от повторов:
    # white_img = np.zeros((400, 200, 3))
        cropped_img = add_image[y1:y2, x1:x2]
        cropped_img = cv2.resize(cropped_img, (200, 400))
        # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        total_img = np.maximum(white_img, cropped_img)
        # plt.imshow(cropped_img)
        # plt.show()

        cv2.imwrite(os.path.join(output_folder, summary_img), total_img)



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
    pose = "pose_side"
    phase = "otn"

    create_summary_image_from_ds(dataset_folder=os.path.join(root, folder_input_imgs, person, pose, phase),
                                 save_name="summary.jpg", person=person, pose=pose, phase=phase)


