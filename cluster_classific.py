import os
import time
import pickle
import cv2
import numpy as np
from typing import Tuple
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from scipy.sparse import  coo_matrix
import matplotlib.pyplot as plt

pose = "pose_fron_left"
svc_path ="svc_{}_150".format(pose)

if not os.path.exists(os.path.join("models", "svc")):
    os.makedirs(os.path.join("models", "svc"))

def calculate_svc(X:np.ndarray, Y:np.ndarray, svc_name:str):
    rbf_svc = SVC(kernel="rbf", gamma=0.03, C=1.0).fit(X, Y)
    with open(os.path.join("models", "svc", svc_name), "wb") as f:
        pickle.dump(rbf_svc, f)
    print("Закончен рассчёт", svc_name)



def plot_svc_results(svc_name:str, X:np.ndarray, Y:np.ndarray, shape:Tuple):

    with open(os.path.join("models", "svc", svc_name), "rb") as f:
        rbf_svc = pickle.load(f)
    try:
        h = 0.5

        x_min, x_max = 0, shape[1]
        y_min, y_max = 0, shape[0]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.scatter(X[:, 1], X[:, 0], c=Y, cmap=plt.cm.coolwarm, s=1)

        Z = Z.reshape(xx.shape)
        plt.contourf(yy, xx, Z, cmap=plt.cm.coolwarm, alpha=0.5)
        plt.axis("on")
        plt.gca().invert_yaxis()
        plt.xlim(yy.min(),yy.max())
        plt.ylim(xx.max(), xx.min())
        plt.show()
    except Exception as e:
        print("Ошибка отрисовки SVC:", e)


if __name__ == '__main__':
    path =os.path.join("output_folder", "first_person", pose, "otn", "summary.jpg")
    img = cv2.imread(path)
    img = np.mean(img, axis=-1)
    img = img.reshape((img.shape[0], img.shape[1]))
    img = np.where(img>75, img, 0)


    sparce = coo_matrix(img)
    points = [el for el in zip(sparce.row, sparce.col)]

    coord_points = points = list(sorted(points, key=lambda x:x[0]))
    coord_points = np.array(coord_points)
    print(coord_points.shape)

    #добавляем DBSCAN для кластеризации и отделения пустого пространства
    grid_xx, grid_yy = np.meshgrid(np.arange(0, img.shape[0], 20),
                                   np.arange(0, img.shape[1], 20))
    grid = np.c_[grid_xx.ravel(), grid_yy.ravel()] # добавляем координаты точек сетки
    all_points = np.concatenate([coord_points, grid])

    for pair in all_points:
        x, y = pair
        img[x][y] = 255

    # plt.imshow(img)
    # plt.show()


    db = DBSCAN(eps=7,  min_samples=3).fit(all_points)
    labels_DBSCAN =db.labels_

    n_clusters = len(set(labels_DBSCAN))- (1 if -1 in labels_DBSCAN else 0)
    print(n_clusters)

    noise_points = all_points[labels_DBSCAN==-1]
    print(noise_points.shape)

    # labels_DBSCAN = np.where(labels_DBSCAN>=0, labels_DBSCAN, 3)
    print(labels_DBSCAN.shape)
    print(all_points.shape)
    calculate_svc(X=all_points, Y=labels_DBSCAN, svc_name=svc_path)
    plot_svc_results(svc_name=svc_path, X=all_points, Y=labels_DBSCAN, shape=(img.shape[1], img.shape[0]))
    # calculate_plot_svc_results(, all_points, labels_DBSCAN, svc_name="aaa", plot=True)

    #### Test results

    # path = os.path.join("input_images", "first_person", "pose_side", "otn", "IMG_1909frame_282.jpg")
    # img = cv2.imread(path)
    # img = np.mean(img, axis=-1)
    # img = img.reshape((img.shape[0], img.shape[1]))
    # img = np.where(img > 75, img, 0)
    #
    # sparce = coo_matrix(img)
    # points = [el for el in zip(sparce.row, sparce.col)]
    #
    # coord_points = points = list(sorted(points, key=lambda x: x[0]))
    # coord_points = np.array(coord_points)
    # print(coord_points.shape)
    #
    # circle1 = plt.Circle((coord_points[0, 1], coord_points[0, 0]), 5, color='r')
    # circle2 = plt.Circle((coord_points[1, 1], coord_points[1, 0]), 5, color='r')
    # circle3 = plt.Circle((coord_points[2, 1], coord_points[2, 0]), 5, color='r')
    # plt.gca().add_patch(circle1)
    # plt.gca().add_patch(circle2)
    # plt.gca().add_patch(circle3)
    #
    # shape =(img.shape[1], img.shape[0])
    # h=0.8
    #
    # with open(os.path.join("models", svc_path), "rb") as f:
    #     rbf_svc = pickle.load(f)
    #
    # x_min, x_max = 0, shape[1]
    # y_min, y_max = 0, shape[0]
    #
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    # plt.scatter(all_points[:, 1], all_points[:, 0], c=labels_DBSCAN, cmap=plt.cm.coolwarm, s=1)
    #
    # Z = Z.reshape(xx.shape)
    # plt.contourf(yy, xx, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    # plt.axis("on")
    # plt.gca().invert_yaxis()
    # plt.xlim(yy.min(), yy.max())
    # plt.ylim(xx.max(), xx.min())
    # plt.show()

