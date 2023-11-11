import cv2
import os

# Подготовка толчка ногой - птн
# Толчковое движение ногой - тдн
# Окончание толчка ногой - отн

queue=['птн','тдн','отн']
# тупой cv2 не может в рускоязычную папку сохранять поэтому нужно это
english=['ptn','tdn','otn']
def read_timecodes(file_txt:str,video: str,save_dir:str):

    # создание папок для сохранения
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    for p in english:
        path=os.path.join(save_dir,p)
        if not (os.path.exists(path)):
            os.mkdir(path)

    # олучение характерисик видео
    cap = cv2.VideoCapture(video)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        fps = round(fps)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = round(fps)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # чтение разметки видео по фазам
    with open(file_txt, encoding = 'utf-8') as f:
        data=f.readlines()

    # находим первую записанную позу
    start=data[0]
    for i in range(3):
        if queue[i] in start:
            start_pose=queue[i]
            start_index=i
            start_frame=int(float(start.split('-')[0])*fps)
            print(start_frame)
            break
    #Индекс строки в которой написана следующая фаза т.е. когда нам надо остановиться
    txt_next_index=1
    stop_frame = int(float(data[txt_next_index].split('-')[0]) * fps)
    #Проход по всем кадрам видео
    num=0
    save_dir_pose=os.path.join(save_dir,start_pose)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        if num>=start_frame:
            # Идет позиция start_pose. Она обновляемая
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            name=os.path.join(save_dir_pose,os.path.basename(video).split('.')[0]+'frame_'+str(num)+'.jpg')

            time=round(num/fps,3)
            org=(image.shape[1]-200,100)
            image = cv2.putText(image, str(time), org, cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0,255,0), 2, cv2.LINE_AA)
            cv2.imwrite(name,image)

            if num>=stop_frame:
                # print("save here:",start_pose,'frame stop saving',stop_frame)
                start_index=(start_index+1)%3
                start_pose=queue[start_index]
                txt_next_index+=1
                try:
                    stop_frame=int(float(data[txt_next_index].split('-')[0])*fps)
                except IndexError:
                    stop_frame=1e10
                english_pose=english[start_index]
                save_dir_pose = os.path.join(save_dir, english_pose)
        num+=1
    print('end function work')
def del_n(value):
    return ''.join(value.splitlines())
def merge_file_txt(file_txt_l:str,file_txt_r:str,file_save:str):
    with open(file_txt_l, encoding = 'utf-8') as f_l:
        data_l=f_l.readlines()
        data_l=[del_n(i) + '_л'  for i in data_l]
    with open(file_txt_r, encoding = 'utf-8') as f_r:
        data_r=f_r.readlines()
        data_r=[del_n(i) +'_п' for i in data_r]
    all=data_l+data_r
    all=sorted(all, key=lambda x:float(x.split('-')[0]))
    all=[i+'\n' for i in all]
    with open(file_save,'w', encoding = 'utf-8') as f_s:
        f_s.writelines(all)

    # all={}
    # for i in data_l:
    #     print
    #     key=float(i.split('-'))[0]
    #     all[key]=i
    # for i in data_r:
    #     key=float(i.split('-'))[0]
    #     all[key]=i
    # print(sorted(all))
def read_timecodes_dl(file_txt:str,video: str,save_dir:str):
    'разметка с двумя ногами'
    # создание папок для сохранения
    queue = ['птн_л', 'тдн_л', 'отн_л','птн_п', 'тдн_п', 'отн_п']
    # тупой cv2 не может в рускоязычную папку сохранять поэтому нужно это
    english = ['ptn l', 'tdn l', 'otn l','ptn r', 'tdn r', 'otn r']
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    for p in english:
        path=os.path.join(save_dir,p)
        if not (os.path.exists(path)):
            os.mkdir(path)

    # олучение характерисик видео
    cap = cv2.VideoCapture(video)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        fps = round(fps)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = round(fps)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # чтение разметки видео по фазам
    with open(file_txt, encoding = 'utf-8') as f:
        data=f.readlines()

    # находим первую записанную позу
    start=data[0]
    for i in range(3):
        if queue[i] in start:
            start_pose=queue[i]
            start_index=i
            start_frame=int(float(start.split('-')[0])*fps)
            print(start_frame)
            break
    #Индекс строки в которой написана следующая фаза т.е. когда нам надо остановиться
    txt_next_index=1
    stop_frame = int(float(data[txt_next_index].split('-')[0]) * fps)
    #Проход по всем кадрам видео
    num=0
    save_dir_pose=os.path.join(save_dir,start_pose)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        if num>=start_frame:
            # Идет позиция start_pose. Она обновляемая
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            name=os.path.join(save_dir_pose,os.path.basename(video).split('.')[0]+'frame_'+str(num)+'.jpg')

            time=round(num/fps,3)
            org=(image.shape[1]-200,100)
            image = cv2.putText(image, str(time), org, cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0,255,0), 2, cv2.LINE_AA)
            cv2.imwrite(name,image)

            if num>=stop_frame:
                print("save here:",start_pose,'frame stop saving',stop_frame)
                start_index=(start_index+1)%3
                start_pose=queue[start_index]
                txt_next_index+=1
                try:
                    stop_frame=int(float(data[txt_next_index].split('-')[0])*fps)
                except IndexError:
                    stop_frame=1e10
                english_pose=english[start_index]
                save_dir_pose = os.path.join(save_dir, english_pose)
        num+=1
    print('end function work')
# read_timecodes(r"D:\hockey\phase_1909.txt",r"D:\hockey\IMG_1909.MOV",r"D:\hockey\pose_side")
# read_timecodes(r"D:\hockey\phase_1904_left.txt",r"D:\hockey\IMG_1904.MOV",r"D:\hockey\pose_fron_left")
# read_timecodes(r"D:\hockey\phase_1904_right.txt",r"D:\hockey\IMG_1904.MOV",r"D:\hockey\pose_fron_right")
# read_timecodes(r"D:\hockey\none.txt",r"D:\hockey\IMG_1904.MOV",r"D:\hockey\all")
# merge_file_txt(r"D:\hockey\phase_1904_left.txt",r"D:\hockey\phase_1904_right.txt",
#                r"D:\hockey\phase_1904_all.txt")
# read_timecodes_dl(r"D:\hockey\phase_1904_all.txt",r"D:\hockey\IMG_1904.MOV",r"D:\hockey\front_all")
read_timecodes(r"D:\hockey\phase_1909.txt",r"D:\hockey\IMG_1909_cut.mp4",r"D:\hockey\pose_side_cut")