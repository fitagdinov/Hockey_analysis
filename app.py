import numpy as np
import os
import re
import cv2
import pickle
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from sklearn import svm


root = os.getcwd()
folder_model = os.path.join(root, "models", "yolov8s-pose.pt")
folder_input_vids = "input_videos"
folder_input_imgs = os.path.join("data", "input_images")
folder_temp_imgs = "temp_images"
folder_output_matrices = "output_folder"




path_to_summary_matrix = os.path.join(os.getcwd(), folder_output_matrices, "summary_matrix.npy")

root = os.getcwd()
yolo_model = YOLO(folder_model)
# for file in os.scandir(folder_input_images):
#     yolo_model(source=file.path, save=save)

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2




def collect_resultimg(person:str, pose:str, phase:str,
                      path_to_images: List[str],
                      points_to_collect:List[int],
                      name_of_summary_matrix:str):

    """Прогнать массив изображений и собрать из них результирующую матрицу по интересующим точкам
    11, 13, 15 - левая нога
    12, 14, 16 - правая нога

    Результирующая матрица сохраняется в выходной папке
    """

    os.makedirs(os.path.join(folder_output_matrices, person), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose, phase), exist_ok=True)
    path_to_summary_matrix = os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix)


    #Результирующая матрица содержит все положения опорных точек

    if os.path.exists(path_to_summary_matrix ):
        summary_matrix = np.load(path_to_summary_matrix)
    else:
        summary_matrix = np.zeros(shape=(400, 200), dtype=np.uint8)

    for path in path_to_images:

        # result = yolo_model(path)#Предсказание
        #
        # # сначала создаём пустое изображение(матрицу), которую заполним нужными точками
        # w, h, _ = result[0].orig_img.shape
        # white_img = np.zeros(shape=(w, h), dtype=np.uint8)#матрица, которая берет себе точки положений
        #
        # zero_box = result[0].boxes.xywh[0]
        # start_pt = (int(zero_box[0 ] -zero_box[2 ] /2), int(zero_box[1 ] -zero_box[3 ] /2)) # x0, y0 - левый верхний угол рамки
        # end_pt = (int(zero_box[0 ] +zero_box[2 ] /2), int(zero_box[1 ] +zero_box[3 ] /2)) #  x1, y1 - правый нижний угол рамки
        #
        # """Замена логики создания результирующего изображения с целью убрать resize и гарантировать точную передачу всех точек"""
        # cooord_pts = []
        # for idx_pt, xy in enumerate(result[0].keypoints.xy[0]): # итерация по точкам для 0-ой рамки
        #     if idx_pt in points_to_collect: # точки, отвечающие за ноги
        #         if xy != [0., 0.]:
        #             white_img[int(xy[1])][int(xy[0])] = 255.
        #             cooord_pts.append([int(xy[0]), int(xy[1])]) # X и Y положения
        #
        # w, h = end_pt[0] - start_pt[0], end_pt[1] - start_pt[1] # размеры ограничивающей рамки , ХУ
        # ratio_w, ratio_h = 200/w, 400/h #коэффициенты ширины и высоты
        # # el - пара точек х и у
        # cooord_pts = list(map(lambda el:[int(ratio_w*(el[0]-start_pt[0])), int(ratio_h*(el[1]-start_pt[1]))], cooord_pts))

        real_img, points_to_draw, points_to_predict, labels = get_predict(path, points_to_collect)

        for xy in points_to_predict:
            if xy[1]<400 and xy[0]<200:
                summary_matrix[xy[1]][xy[0]] = 255.

        ## Проверка себя
        # plt.imshow(summary_matrix)
        # plt.show()


    np.save(path_to_summary_matrix, summary_matrix)




def simulator_many_images(person, pose, phase, name_of_summary_matrix:str, count =50 ):
    """Иммитирует выборку из разных изображений по одному изображению"""

    os.makedirs(os.path.join(folder_output_matrices, person), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose), exist_ok=True)
    os.makedirs(os.path.join(folder_output_matrices, person, pose, phase), exist_ok=True)
    path_to_summary_matrix = os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix)

    summary_matrix = np.load(path_to_summary_matrix)
    img = np.where(summary_matrix > 50, summary_matrix, 0)
    sparse = coo_matrix(img)
    print(sparse)
    points = [el for el in zip(sparse.row, sparse.col)]  # YX по изображению
    points = list(sorted(points, key=lambda x: x[0]))

    labels = []
    coord_points = []
    for i in range(count):
        xx = []
        yy = []
        for idx, pt in enumerate(points):
            y, x = pt
            ofs_y = np.random.randint(-15, 15)
            ofs_x = np.random.randint(-20, 20)
            img[y+ofs_y][x+ofs_x] = 255.
            labels.append(idx)
            coord_points.append([x+ofs_x, y+ofs_y])
            xx.append(x+ofs_x)
            yy.append(y+ofs_y)
    coord_points = np.array(coord_points)

    for xy in coord_points:
        img[xy[1]][xy[0]] = 255.
    # plt.imshow(img)
    # plt.show()

    np.save(os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix), img)

def from_summary_matrix_create_classificator(path_to_summary_matrix:str,
                                             points_to_collect:List[int],
                                             svc_name:str,
                                             epsilon = 10. #гиперпараметр DBSCAN
                                             ):
    """По итоговой матрице распределения положения нужных точек
    строит модель классификации. Всё, что не попало в область положения
    точек заменяется на равномерную сетку точек, таким образом при
    кластеризации мы получаем надкласс 'не попал в точку'
    """
    summary_matrix = np.load(path_to_summary_matrix)

    plt.imshow(summary_matrix, cmap="gray")
    plt.show()

    print(summary_matrix.shape)
    sparse = coo_matrix(summary_matrix)
    coord_points = np.array([el for el in zip(sparse.col, sparse.row)])
    print(coord_points.shape)

    count_classes = len(points_to_collect)+1

    # добавляем DBSCAN для кластеризации и отделения пустого поля
    grid_xx, grid_yy = np.meshgrid(np.arange(0, summary_matrix.shape[1], 15),
                                   np.arange(0, summary_matrix.shape[0], 15))
    grid = np.c_[grid_xx.ravel(), grid_yy.ravel()]  # координаты точек сетки

    all_points = np.concatenate([coord_points, grid])  #Все точки
    print(all_points.shape)

    # Проверка на корректность действий
    white_img = np.zeros(shape=summary_matrix.shape, dtype=np.uint8)
    for xy in all_points:
        if xy[1]<400 and xy[0]<200:
            white_img[xy[1]][xy[0]] = 255.

    plt.imshow(white_img, cmap="gray")
    plt.show()

    # exit()
    #Эпсилон надо как-то подбирать
    db = DBSCAN(eps=epsilon, min_samples=count_classes).fit(all_points)  # кластеризация, 6 класса + 1 "фоновый"
    labels_DBSCAN = db.labels_
    print(set(labels_DBSCAN))

    noise_points = all_points[labels_DBSCAN == -1]  # количество точек "фона"
    print(noise_points.shape)

    all_points_train = np.concatenate([coord_points, noise_points])  # точки сетки, которые определены как фон + целевая выборка
    print(all_points_train.shape) # тут получается меньше, потому что мы удалили часть точек сетки, котоыре попали в область полезных данных

    labels_train = []
    for coord, labl in zip(all_points, labels_DBSCAN):
        if coord in all_points_train:
            labels_train.append(labl)


    shape = summary_matrix.shape
    X = all_points
    Y = labels_train

    h = 0.5 # параметр сетки отрисовки
    rbf_svc = svm.SVC(kernel="rbf", gamma=0.03, C=1.0).fit(X, Y)

    with open(os.path.join("models", "classificators", svc_name), "wb") as f:
        pickle.dump(rbf_svc, f)

    print("Закончен рассчёт", svc_name)

    # Отрисовка для проверки

    x_min, x_max = 0, shape[1]
    y_min, y_max = 0, shape[0]

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print(xx.shape)
    print(yy.shape)

    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=1)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.axis("on")
    plt.gca().invert_yaxis()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.max(), yy.min())

    plt.show()

def get_predict(path:str, points_to_collect:List[int])->Tuple[np.ndarray, List, List, np.ndarray]:
    """Вовзращает кортеж (оригинальное изображение, точки в реальном масштабе, координаты для универсальной рамки(400х200), лейблы точек)"""
    result = yolo_model(path)  # Предсказание

    points_to_draw = []  # точки для отрисовки
    labels = []

    # сначала создаём пустое изображение(матрицу), которую заполним нужными точками
    w, h, _ = result[0].orig_img.shape
    zero_box = result[0].boxes.xywh[0]
    start_pt = (int(zero_box[0] - zero_box[2] / 2), int(zero_box[1] - zero_box[3] / 2))  #
    end_pt = (int(zero_box[0] + zero_box[2] / 2), int(zero_box[1] + zero_box[3] / 2))  #

    """Замена логики распознавания точек для гарантированной связанности лэйблов и координат"""
    for idx_pt, xy in enumerate(result[0].keypoints.xy[0]):  # итерация по точкам для 0-ой рамки
        if idx_pt in points_to_collect:  # точки, отвечающие за ноги
            if xy != [0., 0.]:
                points_to_draw.append([int(xy[0]), int(xy[1])])  # X и Y положения
                labels.append(idx_pt)
    labels = np.array(labels)

    w, h = end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]  # размеры ограничивающей рамки , ХУ
    ratio_w, ratio_h = 200 / w, 400 / h  # коэффициенты ширины и высоты

    # точки для предикта, отмасштабированные, el - пара точек х и у
    points_to_predict = list(map(lambda el: [int(ratio_w * (el[0] - start_pt[0])),
                                             int(ratio_h * (el[1] - start_pt[1]))], points_to_draw))

    return (result[0].orig_img, points_to_draw, points_to_predict, labels)


def predict_on_image(path_to_images:List[str], points_to_collect:List[int], models:List[str]):

    """
    Предсказывает фазу движения по изображению исходя из списка классификаторов
    :param path_to_images: список путей до изображения
    :param points_to_collect: точки, на которые мы ориентируемся
    :param models: список названий классификаторов
    :return:
    """

    for path in path_to_images:
        # result = yolo_model(path)#Предсказани
        #
        # points_to_draw = [] # точки для отрисовки
        # labels = []
        #
        # # сначала создаём пустое изображение(матрицу), которую заполним нужными точками
        # w, h, _ = result[0].orig_img.shape
        # zero_box = result[0].boxes.xywh[0]
        # start_pt = (int(zero_box[0] - zero_box[2] / 2), int(zero_box[1] - zero_box[3] / 2))  #
        # end_pt = (int(zero_box[0] + zero_box[2] / 2), int(zero_box[1] + zero_box[3] / 2))  #
        #
        # """Замена логики распознавания точек для гарантированной связанности лэйблов и координат"""
        # for idx_pt, xy in enumerate(result[0].keypoints.xy[0]): # итерация по точкам для 0-ой рамки
        #     if idx_pt in points_to_collect: # точки, отвечающие за ноги
        #         if xy != [0., 0.]:
        #             points_to_draw.append([int(xy[0]), int(xy[1])]) # X и Y положения
        #             labels.append(idx_pt)
        # labels = np.array(labels)

        real_img, points_to_draw, points_to_predict, labels = get_predict(path, points_to_collect)


        # w, h = end_pt[0] - start_pt[0], end_pt[1] - start_pt[1] # размеры ограничивающей рамки , ХУ
        # ratio_w, ratio_h = 200/w, 400/h #коэффициенты ширины и высоты
        #
        # # точки для предикта, отмасштабированные, el - пара точек х и у
        # points_to_predict = list(map(lambda el:[int(ratio_w*(el[0]-start_pt[0])),
        #                                         int(ratio_h*(el[1]-start_pt[1]))], points_to_draw))

        #
        """Здесь идея в следующем - на вход подаём набор классификаторов, по которым хотим классифицировать
        тот классификатор, который получил наибольшую степень уверенности, выбирается верным
        """
        end_rus ={
            'otn':'окончание толчка ногой',
            'ptn':'подготовка к толчку ногой',
            'tdn':'толчковое движение ногой'
        }

        classificators_for_phases = {el:[end_rus[el[2:5]], 0] for el in models}
        print(classificators_for_phases)

        classificators = [key for key in classificators_for_phases.keys()]
        for draw_idx, classificator in enumerate(classificators):

            with open(os.path.join("models", "classificators", classificator), "rb") as f:
                rbf_svc = pickle.load(f)

            predicted_labels = rbf_svc.predict(points_to_predict)+min(points_to_collect)

            # # Проверка на правильность
            # full_phase = os.path.dirname(path)
            # phase = os.path.basename(full_phase)
            # full_pose = os.path.dirname(full_phase)
            # pose = os.path.basename(full_pose)
            # full_person = os.path.dirname(full_pose)
            # person = os.path.basename(full_person)
            #
            # for pts, lab_p, lab_t in zip(points_to_predict, predicted_labels, labels):
            #     print("{} | {}(true = {})".format(pts, lab_p, lab_t))
            #
            # img = np.zeros(shape=(400, 200), dtype=np.uint8)
            # for xy in points_to_predict:
            #     img[xy[1]][xy[0]] = 255.
            # path_to_summary_matrix = os.path.join(output_folder, person, pose, phase,
            #                                       "summary_matrix.npy")
            # summary = np.load(path_to_summary_matrix)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(summary)
            # ax[1].imshow(img)
            # plt.show()
            # # exit()

            conf = len(labels[labels==predicted_labels])/len(labels)
            print("conf lvl for {} = {}".format(classificators_for_phases[classificator][0], conf))
            classificators_for_phases[classificator][1] = conf

        # Добавляем информацию на оригинальное изображение
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        for i, xy, lab in zip(list(range(len(labels))), points_to_draw, labels):
            real_img = cv2.putText(real_img,  str(lab),  (int(xy[0]), int(xy[1])), font,
                                                            2, (0,0,255), thickness, cv2.LINE_AA)
        max_score = 0
        predicted_phase =""
        for k,v in classificators_for_phases.items():
            if v[1]>max_score:
                max_score = v[1]
                predicted_phase = v[0]

        # real_img = cv2.putText(real_img, "Фаза движения: {}".format(predicted_phase), (50, 50), font,
        #                        1, (0,0,255), thickness, cv2.LINE_AA)

        plt.imshow(real_img)
        plt.title("Фаза движения: {}".format(predicted_phase))
        plt.axis(False)
        plt.show()


if __name__ == '__main__':

    person = "first_person"
    pose = "pose_side"
    phase = "tdn"
    image_folder = os.path.join(folder_input_imgs, person, pose, phase)
    images = [os.path.join(image_folder, file) for file in os.listdir(image_folder)][:1]

    name_of_summary_matrix = "summary_matrix.npy"
    points_to_collect = [13, 14, 15, 16]
    # Собрать результирующую матрицу
    # collect_resultimg(person=person, pose=pose, phase=phase, path_to_images=images,
    #                   points_to_collect=points_to_collect,
    #                   name_of_summary_matrix=name_of_summary_matrix,
    #                   )
    # # # Симулятор сбора, при отсутсвии выборки
    # simulator_many_images(person=person, pose=pose, phase=phase,
    #                       name_of_summary_matrix=name_of_summary_matrix,
    #                       count=150)

    # # Делаем классификатор на основе тренировочных данных
    # from_summary_matrix_create_classificator(path_to_summary_matrix = os.path.join(folder_output_matrices, person, pose, phase, name_of_summary_matrix),
    #                                          points_to_collect=points_to_collect,
    #                                          svc_name='{}_{}_side_model_svc'.format(len(points_to_collect),phase),
    #                                          epsilon=10.)

    person = "random_persons_for_presentation"
    pose = "front"
    phase = "ptn"
    image_folder = os.path.join(folder_input_imgs, person, pose, phase)
    images = [os.path.join(image_folder, file) for file in os.listdir(image_folder)][:50]
    #
    predict_on_image(images,
                     points_to_collect=points_to_collect,
                     # models=['4_ptn_fron_model_svc', '4_tdn_fron_model_svc', '4_otn_fron_model_svc']
                     models = [el for el in os.listdir(os.path.join("models", "classificators"))]
                     )
