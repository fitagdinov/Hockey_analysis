import cv2
import numpy as np
import json_to_points
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import show_points
import os
# os.chdir("..")


# TODO create utils and add this function
def int_points(l):
    x,y,p=l
    return([int(x),int(y),p])
def angle_of_vectors(a, b, c, d):
    dotProduct = a * c + b * d
    # for three dimensional simply add dotProduct = a*c + b*d  + e*f
    modOfVector1 = math.sqrt(a * a + b * b) * math.sqrt(c * c + d * d)
    # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f)
    angle = dotProduct / modOfVector1
    angleInDegree = math.degrees(math.acos(angle))
    return angleInDegree

def len_vec(vec):
    return (vec[0]**2+vec[1]**2)**0.5
# TODO( в один файл закинуть этот список иоттуда читать. Во всех файлах он вначале)

# igor's points

# points=['nose', 'right_eye', 'left_eye', 'right_ear',
#           'left_ear', 'right_shoulder', 'left_shoulder',
#           'right_elbow', 'left_elbow', 'right_brush',
#           'left_brush', 'right_hip', 'left_hip', 'right_knee',
#           'left_knee', 'right_ankle', 'left_ankle', 'breast',
#           'top', 'right_big_toe', 'right_small_toe', 'right_knee',
#           'left_big_toe', 'left_small_toe', 'left_heel']

# robert's points
points=['chin',"breast",'left_shoulder','left_elbow','left_brush','right_shoulder','right_elbow','right_brush',
            'groin','left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle','left_eye','right_eye',
            'left_ear','right_ear','right_foot_mid','right_foot_front',
           'right_foot_back','left_foot_mid','left_foot_front','left_foot_back']
def angles(image,dict_points,show_res=True,part=1,opr_leg=1):
    global points
    # получение точек and отрисовка их

    # dict_points = json_to_points.json_to_points(file_point_old)

    # image=show_points.on_image(dict_points, image, points=points)
    # отрисовка вертикали


    # TODO(разобраться с точками и вставить нормалльное обозначение)
    hip1_x,hip1_y,hip1_p=dict_points[points[11]]
    hip2_x, hip2_y, hip2_p = dict_points[points[12]]
    groin_x, groin_y= int((hip1_x+hip2_x)/2),int((hip1_y+hip2_y)/2)

    sh1_x, sh1_y, sh1_p = dict_points[points[6]]
    sh2_x, sh2_y, sh2_p = dict_points[points[5]]

    br_x, br_y = int((sh1_x + sh2_x) / 2), int((sh1_y + sh2_y) / 2)

    # TODO( нормальную функцию написать: угол между двумя прямыми с координатами 2х точек на каждой прямой)
    gr_vec=np.array([groin_x,0])-np.array([groin_x,groin_y])
    br_vec=np.array([br_x,br_y])-np.array([groin_x,groin_y])
    angle_body=angle_of_vectors(br_vec[0],br_vec[1],gr_vec[0],gr_vec[1])
    # print('angle back:',angle_body)

    # отрисовка угла колена
    hip1_x, hip1_y, hip1_p = int_points(dict_points[points[11]])
    hip2_x, hip2_y, hip2_p = int_points(dict_points[points[12]])
    knee1_x, knee1_y, knee1_p = int_points(dict_points[points[13]])
    knee2_x, knee2_y, knee2_p = int_points(dict_points[points[14]])
    ankle1_x, ankle1_y, ankle1_p = int_points(dict_points[points[15]])
    ankle2_x, ankle2_y, ankle2_p = int_points(dict_points[points[16]])
# TODO(преепутано верх и низ)
    down_leg_vec_1 = np.array([knee1_x, knee1_y]) - np.array([ankle1_x, ankle1_y])
    up_leg_vec_1 = np.array([knee1_x, knee1_y]) - np.array([hip1_x, hip1_y])

    down_leg_vec_2 = np.array([knee2_x, knee2_y]) - np.array([ankle2_x, ankle2_y])
    up_leg_vec_2 = np.array([knee2_x, knee2_y]) - np.array([hip2_x, hip2_y])

    # горизонтальная
    angle_leg_1 = angle_of_vectors(up_leg_vec_1[0], up_leg_vec_1[1], down_leg_vec_1[0], down_leg_vec_1[1])

    angle_leg_2 = angle_of_vectors(up_leg_vec_2[0], up_leg_vec_2[1], down_leg_vec_2[0], down_leg_vec_2[1])

    # вертикальная
    angle_leg_1=np.arccos(len_vec(up_leg_vec_1)/len_vec(down_leg_vec_1))/3.14*180
    angle_leg_2=np.arccos(len_vec(up_leg_vec_2)/len_vec(down_leg_vec_2))/3.14*180

    # if opr_leg=1:
    #     #right
    #     opr_knee=int_points(dict_points[points[14]])
    # else:
    #     #left
    #     opr_knee = int_points(dict_points[points[13]])
    #
    # dist_knee_vert=opr_knee[0]- groin_x


    if show_res:
        image = cv2.line(image, (groin_x, groin_y), (groin_x, 0), (0, 255, 0), thickness=2)
        image = cv2.line(image, (groin_x, groin_y), (br_x, br_y), (0, 255, 0), thickness=2)

        image = cv2.line(image, (hip1_x, hip1_y), (knee1_x, knee1_y), (255, 255, 0), thickness=2)
        image = cv2.line(image, (ankle1_x, ankle1_y), (knee1_x, knee1_y), (255, 255, 0), thickness=2)

        image = cv2.line(image, (hip2_x, hip2_y), (knee2_x, knee2_y), (0, 255, 255), thickness=2)
        image = cv2.line(image, (ankle2_x, ankle2_y), (knee2_x, knee2_y), (0, 255, 255), thickness=2)

        string='angle back:   '+ str(round(angle_body,1))

        ord_y=image.shape[0]-300
        ord_x=image.shape[1]-550
        image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)

        string = 'angle_leg_right:      ' + str(round( angle_leg_1,1))

        ord_y = image.shape[0] - 260
        ord_x = image.shape[1] - 550
        image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)

        string = 'angle_leg_left:      ' + str(round( angle_leg_2,1))

        ord_y = image.shape[0] - 230
        ord_x = image.shape[1] - 550
        image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0,0), 1, cv2.LINE_AA)


    return image

def leg(image,dict_points,show_res=True,part=1,opr_leg=1):
    hip1_x, hip1_y, hip1_p = dict_points[points[11]]
    hip2_x, hip2_y, hip2_p = dict_points[points[12]]
    groin_x, groin_y = int((hip1_x + hip2_x) / 2), int((hip1_y + hip2_y) / 2)
    foot1_x, foot1_y, foot1_p = int_points(dict_points[points[19]])
    foot2_x, foot2_y, foot2_p = int_points(dict_points[points[22]])
    if (foot1_x-groin_x)*(foot2_x-groin_x)<0:
        res='good'
    else:
        res='bad'
    if show_res:
        string='the location of the legs is ' + res
        ord_y = image.shape[0] - 330
        ord_x = image.shape[1] - 550
        image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return image
def elbow(image,dict_points,show_res=True,part=1,opr_leg=1):
    hip1_x, hip1_y, hip1_p = dict_points[points[11]]
    hip2_x, hip2_y, hip2_p = dict_points[points[12]]
    groin_x, groin_y = int((hip1_x + hip2_x) / 2), int((hip1_y + hip2_y) / 2)
    elbow1_x, elbow1_y, elbow1_p = int_points(dict_points[points[7]])
    elbow2_x, elbow2_y, elbow2_p = int_points(dict_points[points[8]])
    if (elbow1_x-groin_x)*(elbow2_x-groin_x)<0:
        res='good'
    else:
        res='bad'
    if show_res:
        string='the location of the elbow is ' + res
        ord_y = image.shape[0] - 360
        ord_x = image.shape[1] - 550
        image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return image