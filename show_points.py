import cv2
import matplotlib.pyplot as plt
import json
import os
import json_to_points
colors_file=r'other/colors.json'
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



def on_image(dict_points,image,points):
    # global colors
    # TODO( file error if use it in other directory. ex. side_camera)
    with open(colors_file, 'r') as f:
        colors = json.load(f)
    i=0
    for p in points:
        x,y,probably= dict_points[p]
        #TODO переход из 16 бит в РГБ косой. Надо чтобы cv2 понимла 16 бит или переписать файл
        color=colors[p][1:]
        color=tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        if (x,y)!=(0.0,0.0):
            # print(x,y)
            # print(color)
            image=cv2.circle(image, (int(x),int(y)), radius=2, color=color, thickness=2)
            # отрисовка названия точки
            image=cv2.putText(image,str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 1, cv2.LINE_AA)
        else:
            print(p,"  don't find")
        i+=1
    return image
def all_image(path_image,path_dict):
    #TODO Прописать сохраниение и визуализацию по флагу в аргументах функции
    # TODO отрисовку необходиых линий
    for n in os.listdir(path_image):
        name=os.path.join(path_image,n)
        image=cv2.imread(name)
        # номер кадра
        #TODO( сделать реализацию получше)
        num_frame=n.split('.')[0].split('_')[-1]
        file_list=os.listdir(path_dict)
        # shablon='IMG_1909_000000000000_keypoints.json'
        # print(file_list[0].split['_'])
        file_num_rand=file_list[0].split('_')[-2]
        file_num_=file_num_rand[:-len(num_frame)]+num_frame
        file_point_old=file_list[0].replace(file_num_rand,file_num_)
        file_point_old=os.path.join(path_dict,file_point_old)
        dict_points = json_to_points.json_to_points(file_point_old)
        new_image = on_image(dict_points, image, points=points)
        cv2.imshow('show', new_image)
        cv2.waitKey()

# all_image(path_image=r'D:\hockey\pose\tdn',path_dict=r'D:\hockey\json_1909\json_1909')