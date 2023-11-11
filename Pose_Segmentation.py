import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

vid = cv2.VideoCapture("D:\hockey\IMG_1568.MOV")

def Segmentation(cap):
    with mp_pose.Pose(
        min_detection_confidence=0.05,
        min_tracking_confidence=0.05,
        enable_segmentation=True) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            
            BG_COLOR = (192, 192, 192) # gray

            # apply mask on image with gray
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            image = np.where(condition, image, bg_image)
            image_height, image_width, _ =  image.shape
            image = cv2.resize(image, (int(image_width * (640 / image_height)), 640))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

    return

def Detection(cap):
    with mp_pose.Pose(
        min_detection_confidence=0.05,
        min_tracking_confidence=0.05,
        enable_segmentation=True) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            image_height, image_width, _ =  image.shape
            image = cv2.resize(image, (int(image_width * (640 / image_height)), 640))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    return

Detection(vid)
# Segmentation(vid)
