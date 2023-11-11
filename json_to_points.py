import json

# igor's points

# keyList = ['nose', 'right_eye', 'left_eye', 'right_ear',
#           'left_ear', 'right_shoulder', 'left_shoulder',
#           'right_elbow', 'left_elbow', 'right_brush',
#           'left_brush', 'right_hip', 'left_hip', 'right_knee',
#           'left_knee', 'right_ankle', 'left_ankle', 'breast',
#           'top', 'right_big_toe', 'right_small_toe', 'right_knee',
#           'left_big_toe', 'left_small_toe', 'left_heel']


#robert's points
keyList=['chin',"breast",'left_shoulder','left_elbow','left_brush','right_shoulder','right_elbow','right_brush',
            'groin','left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle','left_eye','right_eye',
            'left_ear','right_ear','right_foot_mid','right_foot_front',
           'right_foot_back','left_foot_mid','left_foot_front','left_foot_back']

def json_to_points(file:str):
    '''
    Функция преобразыет json файл полученый из openpose в словарь с подписаными частями тела
    выходной json имеет 75 точек идущих подряд: х кордината, у кордината, вероятность
    :param file - путь до json файла:
    :return dict  словарь с подписанными частями тела :
    '''
    global keyList
    points_dict = {}
    with open(file,'r') as f:
        data = json.load(f)
        points=data['people'][0]['pose_keypoints_2d']# неразмеченные точки
    for i in range(len(keyList)):
        point=points[3*i:3*i+3]
        points_dict[keyList[i]]=point
    return points_dict
def save_new_json(file: str, save_file : str):
    '''
    Функция для сохранения результатов функции json_to_points в виде json
    :param file файл чтения json файла из openpose:
    :param save_file Путь для сохранения нового файла.
    ЕСЛИ file == save_file ФАЙЛ ПЕРЕЗАПИШЕТСЯ. БУДТЕ АККУРАТНЫ:
    :return:
    '''
    points_dict=json_to_points(file)
    json_string = json.dumps(points_dict)
    with open(save_file,'w') as save:
        save.write(json_string)
