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
def angles(image,dict_points,ph,show_res=True):
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

    up_leg_vec_1 = np.array([knee1_x, knee1_y]) - np.array([ankle1_x, ankle1_y])
    down_leg_vec_1 = np.array([knee1_x, knee1_y]) - np.array([hip1_x, hip1_y])

    up_leg_vec_2 = np.array([knee2_x, knee2_y]) - np.array([ankle2_x, ankle2_y])
    down_leg_vec_2 = np.array([knee2_x, knee2_y]) - np.array([hip2_x, hip2_y])


    angle_leg_1 = angle_of_vectors(up_leg_vec_1[0], up_leg_vec_1[1], down_leg_vec_1[0], down_leg_vec_1[1])
    # print('angle_leg_left:', 180-angle_leg_1)

    angle_leg_2 = angle_of_vectors(up_leg_vec_2[0], up_leg_vec_2[1], down_leg_vec_2[0], down_leg_vec_2[1])
    # print('angle_leg_right:', 180-angle_leg_2)

    if show_res:

        image = cv2.line(image, (groin_x, groin_y), (groin_x, 0), (0, 255, 0), thickness=2)
        image = cv2.line(image, (groin_x, groin_y), (br_x, br_y), (0, 255, 0), thickness=2)

        image = cv2.line(image, (hip1_x, hip1_y), (knee1_x, knee1_y), (255, 255, 0), thickness=2)
        image = cv2.line(image, (ankle1_x, ankle1_y), (knee1_x, knee1_y), (255, 255, 0), thickness=2)

        image = cv2.line(image, (hip2_x, hip2_y), (knee2_x, knee2_y), (0, 255, 255), thickness=2)
        image = cv2.line(image, (ankle2_x, ankle2_y), (knee2_x, knee2_y), (0, 255, 255), thickness=2)

        string='angle back:   '+ str(round(angle_body,1)) + ' Good - not more 10'

        ord_y=image.shape[0]-300
        ord_x=image.shape[1]-550
        image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if ph == -1:
            support_leg=round( angle_leg_1,1)
            not_suport_leg=round( angle_leg_2,1)
            string = 'support leg:      ' + str(round( angle_leg_1,1))

            ord_y = image.shape[0] - 260
            ord_x = image.shape[1] - 550
            image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)

            string = 'not support leg:      ' + str(round( angle_leg_2,1))

            ord_y = image.shape[0] - 230
            ord_x = image.shape[1] - 550
            image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0,0), 1, cv2.LINE_AA)

        else:
            support_leg = round(angle_leg_2, 1)
            not_suport_leg = round(angle_leg_1, 1)
            string = 'not support leg:      ' + str(round(angle_leg_1, 1))

            ord_y = image.shape[0] - 260
            ord_x = image.shape[1] - 550
            image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)

            string = 'support leg:      ' + str(round(angle_leg_2, 1))

            ord_y = image.shape[0] - 230
            ord_x = image.shape[1] - 550
            image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return (image,support_leg)

def foot_angle(top,down):
    x_top,y_top,p_top=int_points(top)
    x_down,y_down, p_down=int_points(down)
    vec1=np.array([x_down, x_down]) - np.array([x_top, y_top])
    vec2=np.array([x_down, 0])
    return angle_of_vectors(vec1[0],vec1[1],vec2[0],vec2[1])


def opor_leg_metric(list_image,list_file_point_old):
    global points
    left_angles=np.zeros(len(list_image))
    right_angles = np.zeros(len(list_image))
    if len(list_image)>10:
        for i in range(len(list_image)):
            image=list_image[i]
            file_point_old=list_file_point_old[i]
            dict_points = json_to_points.json_to_points(file_point_old)
            left_foot_top=dict_points[points[19]]
            left_foot_down = dict_points[points[21]]
            right_foot_top=dict_points[points[22]]
            right_foot_down = dict_points[points[24]]
            left_angles[i]=left_foot_top[0]
            right_angles[i]=right_foot_top[0]
            # left_foot_angle=foot_angle(left_foot_top,left_foot_down)
            # right_foot_angle=foot_angle(right_foot_top,right_foot_down)
            # left_angles[i]=left_foot_angle
            # right_angles[i]=right_foot_angle


        # linel regration
        res_left = stats.linregress(np.arange(left_angles.shape[0]),left_angles)
        res_right = stats.linregress(np.arange(right_angles.shape[0]), right_angles)

        # Coefficient of determination (R-squared)
        R_l=res_left.rvalue ** 2
        R_r = res_right.rvalue ** 2

        R_mean=(R_r+R_l)/2
        # reg = LinearRegression()
        # reg.fit(np.arange(left_angles.shape[0]),left_angles)



        return(R_mean,res_left.slope,res_right.slope)

    else:
        return(0,np.NAN,np.NAN)


        # plt.figure()
        # plt.plot(right_angles,'r')
        # plt.plot(left_angles,'g')
        # x=np.arange(left_angles.shape[0])
        # plt.plot(x, res_left.intercept + res_left.slope*x, '--g')
        # x = np.arange(right_angles.shape[0])
        # plt.plot(x, res_right.intercept + res_right.slope * x, '--r')
        # plt.legend(['right','left','R_left ' + str(R_l),'R_right ' + str(R_r)])
        # plt.show()
        # # for check developer. delete after
        # name=r'D:\hockey\plots_foots'
        # inp=input()
        # name=os.path.join(name,inp+'.jpg')
        # plt.savefig(name)
def define_opr_leg(path_image,path_dict):
    list_image = []
    list_points = []
    last_num_image = 0

    R_mean_list=[]
    left_slope_list=[]
    right_slope_list=[]

    names = os.listdir(path_image)
    names = sorted(names, key=lambda i: int(i.split('.')[0].split('_')[-1]))
    for n in names:
        name = os.path.join(path_image, n)
        num_frame = n.split('.')[0].split('_')[-1]
        file_list = os.listdir(path_dict)
        file_num_rand = file_list[0].split('_')[-2]
        file_num_ = file_num_rand[:-len(num_frame)] + num_frame
        file_point_old = file_list[0].replace(file_num_rand, file_num_)
        file_point_old = os.path.join(path_dict, file_point_old)

        # for opr leg
        num = int(name.split('.')[-2].split('_')[-1])
        if num - last_num_image > 1:
            R_mean,left_slope,right_slope=opor_leg_metric(list_image,list_file_point_old=list_points)
            list_image = []
            list_points = []
            R_mean_list.append(R_mean)
            left_slope_list.append(left_slope)
            right_slope_list.append(right_slope)
        else:
            list_image.append(name)
            list_points.append(file_point_old)
        last_num_image = num

    # нахождение максимально точного предсказания
    R_mean_list=np.array(R_mean_list)
    index=np.argmax(R_mean_list)

    phases=np.zeros(R_mean_list.shape[0]+1)

    # нога правая опрная -> 1 ; нога левая опорная -> -1
    #TODO(сделать проверку на раззные наклоны)
    if (right_slope_list[index]<0 and left_slope_list[index]>0):
        phases[index]=-1
    elif (right_slope_list[index]>0 and left_slope_list[index]<0):
        phases[index]=1
    else:
        print(" Я КОД И Я СЛОМАЛСЯ. АЙ-АЙ-АЙ")
        pass

    #TODO(возможно сделать нормальный проход)
    for i in range(index+1,phases.shape[0]):
        phases[i]=(-1)*phases[i-1]
    for i in range(index-1,-1,-1):
        phases[i] = (-1) * phases[i + 1]
    return phases

def write_opr_leg(image,dict_points,ph,show_res=True):
    if ph==1:
        big_toe=int_points(dict_points[points[22]])
        knee=int_points(dict_points[points[14]])
    else:
        big_toe=int_points(dict_points[points[19]])
        knee = int_points(dict_points[points[13]])
    nose=int_points(dict_points[points[0]])
    image = cv2.line(image, (big_toe[0], big_toe[1]), (knee[0], knee[1]), (125, 255, 125), thickness=2)
    image = cv2.line(image, (nose[0], nose[1]), (knee[0], knee[1]), (125, 255, 125), thickness=2)
    vec_1 = np.array([knee[0], knee[1]]) - np.array([big_toe[0], big_toe[1]])
    vec_2 = np.array([knee[0], knee[1]]) - np.array([nose[0], nose[1]])
    angle = angle_of_vectors(vec_1[0], vec_1[1], vec_2[0], vec_2[1])
    if show_res:
        string='angle nose:   '+ str(round(angle,1))

        ord_y=image.shape[0]-350
        ord_x=image.shape[1]-550
        image = cv2.putText(image, string, (ord_x, ord_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return (image,round(angle,1))





