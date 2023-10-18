
from typing import Any
import cv2
import time

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time

import random
import cv2
import time
from engine.predict import Face2dKeypointPredictor




class OperateDetect(object):

    def __init__(self):
        self.weight = "./shape_predictor_68_face_landmarks.dat"
        self.EYE_AR_THRESH = 0.25  # 眼睛对应的长宽比低于该阈值算一次眨眼
        self.CONSEC_FRAMES = 5  # 这个值被设置为 3，表明眼睛长宽比小于3时，接着三个连续的帧一定发生眨眼动作
        self.COUNTER = 0  # 眼图长宽比小于EYE_AR_THRESH的连续帧的总数
        self.TOTAL = 0  # 脚本运行时发生的眨眼的总次数
        self.MOUTH_AR_THRESH = 0.8
        self.Nod_threshold = 0.03
        self.shake_threshold = 0.03


        self.predictor = Face2dKeypointPredictor()

        self.ACTIONS = ['nothing', 'blink', 'gape', 'shake', 'nod']

        self.mStart, self.mEnd = 84, 96 # 嘴部对应的索引
        self.nStart, self.nEnd = 51, 65 #鼻子对应的索引
        self.lStart, self.lEnd = 75, 82 #左眼对应的的索引
        self.rStart, self.rEnd = 66, 73 #右眼对应的索引
        self.compare_point = [0, 0] # 刚开始的时候设置鼻子的中点在【0， 0】


    def __call__(self, images, action):
        result = {}
        keypoints = self.predecte(images)
        is_live, is_error = self.detect_action(keypoints, action)
        result['is_error'] = is_error
        result['alive'] = is_live
        result['action'] = self.ACTIONS[action]
        
        
        

        return result
    
    def eye_aspect_ratio(self, eye):
        '''
        A，B是计算两组垂直眼睛标志之间的距离，而C是计算水平眼睛标志之间的距离
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        :param eye:两个眼睛在图片中的像素坐标
        :return:返回眼睛的长宽比
        '''

        # A = dist.euclidean(eye[1], eye[5])
        # B = dist.euclidean(eye[2], eye[4])
        # C = dist.euclidean(eye[0], eye[3])
        # ear = (A + B) / (2.0 * C)  # 将分子和分母相结合，得出最终的眼睛纵横比。然后将眼图长宽比返回给调用函数
        A = dist.euclidean(eye[2], eye[6])
        B = dist.euclidean(eye[0], eye[4])
        ear = A / B
        return ear

    def mouth_aspect_ratio(self, mouth):
        # 计算两组垂直方向的欧式距离
        A = dist.euclidean(mouth[3], mouth[9])  # 51, 59
        B = dist.euclidean(mouth[14], mouth[18])  # 53, 57
        # 计算水平方向的距离
        C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
        D = dist.euclidean(mouth[12], mouth[16])
        mar = (A + B) / (C + D)
        return mar

    def center_point(self, nose):
        return nose.mean(axis=0)


    def nod_aspect_ratio(self, size, pre_point, now_point):
        return abs(float((pre_point[1] - now_point[1]) / (size[0] / 2)))


    def shake_aspect_ratio(self, size, pre_point, now_point):
        return abs(float((pre_point[0] - now_point[0]) / (size[1] / 2)))


    def blinks_detect(self, keypoint):
        leftEye = keypoint[self.lStart:self.lEnd]
        rightEye = keypoint[self.rStart:self.rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        if ear < self.EYE_AR_THRESH:
            return True 
        
        # 计算左右眼的凸包，并可视化
        # leftEyeHull = cv2.convexHull(leftEye)
        # rightEyeHull = cv2.convexHull(rightEye)
        return False


    def mouth_detect(self, shape):
        mouth = shape[self.mStart:self.mEnd]

        mouthMAR = self.mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)
        return mar, mouthHull


    def nod_shark(self, size, shape):
        nose = shape[self.nStart:self.nEnd]
        nose_center = self.center_point(nose)
        nod_value, shake_value = 0, 0
        noseHull = cv2.convexHull(nose)
        if  self.compare_point[0] != 0:
            nod_value = self.nod_aspect_ratio(size, nose_center, self.compare_point)
            shake_value = self.shake_aspect_ratio(size, nose_center, self.compare_point)
        self.compare_point = nose_center
        return nod_value, shake_value, noseHull


    def in_area(self, key_points, circle_center, radius):
        
        return True


    def action_judgment(self, action_value):
        action_type = np.array([0, 0, 0, 0])
        ear, mar, nod_value, shake_value = action_value
        if ear < self.EYE_AR_THRESH:
            action_type[0] = 1
        if mar > self.MOUTH_AR_THRESH:
            action_type[1] = 1
        if nod_value > self.Nod_threshold:
            action_type[2] = 1
        if shake_value > self.shake_threshold:
            action_type[3] = 1
        return action_type
    def predecte(self, images):
        keypoints = []
        for frame in images:
            keypoints.append(self.predictor(frame))
        return keypoints

    def detect_action(self, keypoints, action):
        is_live = False
        is_error = False

        in_circle = self.in_area(keypoints, np.array([320, 180]), 90)
        if not in_circle:
            is_error = True
            return is_live, is_error
        
        if action == 1:
            # blink
            for keypoint in keypoints:
                is_live = self.blinks_detect(keypoint)
                if is_live:
                    break

        elif action == 2:
            # gape
            # is_live = 
            pass
        elif action ==3:
            # shake
            pass
        elif action == 4:
            # nod
            pass

        return is_live, is_error

        # eyes_open = self.blinks_detect(keypoints)
        # mar, mouthHull = self.mouth_detect(keypoints)
        # nod_value, shake_value, noseHull = self.nod_shark(size, keypoints)
        # act_type = self.action_judgment((ear, mar, nod_value, shake_value))
        # return act_type, leftEyeHull, rightEyeHull, mouthHull, noseHull


def test():
    live_detector = OperateDetect()
    frame = cv2.imread('./test.jpg')
    #status:[1,2,3,4] eyes,mouth,sheck,nod
    is_live = live_detector([frame],1)
    print(is_live)





# def main():
#     vs = VideoStream(src=0).start()
#     live_detect = OperateDetect()
#     # 每次运行的时候做三中方式的检测， 顺序打乱

#     detect_type = [0, 1, 2, 3]
#     random.shuffle(detect_type)
#     detect_type = detect_type[0:-1]
#     frame = vs.read()
#     frame = imutils.resize(frame, width=640)
#     size = frame.shape
#     activate_judge = np.array([0, 0, 0, 0])
#     first_frame_type = True
#     while True:
#         frame = vs.read()
#         frame = imutils.resize(frame, width=640)
#         cv2.circle(frame, (int(size[1] / 2), int(size[0] / 2)), int(min(size[0] / 2, size[1] / 2) * 0.5), (0, 0, 255))
#         detect_result = live_detect.detect(frame)
#         operate_step = True
#         if len(detect_type) ==0:
#             break
#         if detect_result:
#             if(len(detect_result)) ==6:
#                 is_align, act_info, leftEye, rightEye, mouth_point, nose_point = detect_result
#                 detect_num = detect_type[0]
#                 if first_frame_type:
#                     if not is_align:
#                         operate_step = False
#                 if operate_step:
#                     if detect_num == 0:
#                         cv2.putText(frame, "Trouble blinking", (200, 100),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#                     if detect_num == 1:  # 张嘴操作
#                         cv2.putText(frame, "Open Mouth", (200, 100),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     if detect_num == 2:  # 点头
#                         cv2.putText(frame, "Nod", (200, 100),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#                     if detect_num == 3:  # 摇头操作
#                         cv2.putText(frame, "Shake head", (200, 100),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     if is_align:
#                         first_frame_type = False

#                     if act_info[detect_num] == 1:
#                         activate_judge = activate_judge + act_info
#                         if activate_judge[detect_num] > 3:
#                             cv2.putText(frame, "Single test completed!", (30, 60),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                             print("detect_type[0]", detect_type[0])
#                             detect_type.remove(detect_type[0])
#                             first_frame_type = True

#                     else:
#                         activate_judge = np.array([0, 0, 0, 0])
#                     cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
#                     cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)
#                     cv2.drawContours(frame, [mouth_point], -1, (0, 255, 0), 1)
#                     cv2.drawContours(frame, [nose_point], -1, (0, 255, 0), 1)
#                 else:
#                     cv2.putText(frame, "Trouble align your face in circle", (200, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if len(detect_type) == 0:
#             break

#     cv2.destroyAllWindows()
#     vs.stop()


# if __name__=='__main__':
#     video_captur = cv2.VideoCapture(0)
#     while True:
#         ret, frame = video_captur.read()

#         # cv2.imshow(frame)
#         cv2.imshow('test.jpg', frame)
#         # time.sleep(1)
#         cv2.waitKey(100)
if __name__ == '__main__':
    test()