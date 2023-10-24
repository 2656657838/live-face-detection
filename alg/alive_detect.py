
from scipy.spatial import distance as dist
import cv2
import torch
import numpy as np

import kornia

from core.logger import LOGGER as logger
from core.config import settings
from engine import *



#################################################
# test
#############################################
import matplotlib.pyplot as plt
def plot(data,name):
    plt.plot(data, marker='o', linestyle='-')

# 添加标题和标签
    plt.title('折线图')
    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')

    # 显示折线图
    plt.savefig(f'{name}.jpg')
    plt.clf()
###################################################
class OperateDetect(object):

    def __init__(self):

        self.EYE_AR_THRESH = settings.EYE_AR_THRESH

        self.EYE_MIN = settings.EYE_MIN

        self.MOUTH_AR_THRESH = settings.MOUTH_AR_THRESH

        self.PITCH_MAX = settings.PITCH_MAX
        self.PITCH_MIN = settings.PITCH_MIN
        self.YAW_MAX = settings.YAW_MAX
        self.YAW_MIN = settings.YAW_MIN

        self.mStart, self.mEnd = settings.mStart, settings.mEnd
        self.nStart, self.nEnd = settings.nStart, settings.nEnd
        self.lStart, self.lEnd = settings.lStart, settings.lEnd
        self.rStart, self.rEnd = settings.rStart, settings.rEnd
        self.lbrow, self.rbrow = settings.lBrow, settings.rBrow
    
        self.img_shape = None

        self.face_2d_kpts = face_2d_kpts

        self.ACTIONS = ['nothing', 'blink', 'gape', 'shake', 'nod']


    def __call__(self, images, action):
        result = {
            'action': action,
            'alive': False,
            'conf': None,
            'is_error': False,
            'msg': None
        }
        
        self.img_shape = images[0].shape

        if action == self.ACTIONS[0]:


            alive = False
            conf = None
            alive, conf = self.detect_anti_face(images)
            result['alive'] = alive
            result['conf'] = conf
            return result
        try:
            keypoints, poses, boxes = self.predecte(images)
            if len(keypoints) == 0:
                logger.info('No face detected!')
                result['msg'] = 'No face detected!'
                return result
        except Exception as e:
            result['is_error'] = True
            result['msg'] = f'predecte error: {e}'
            return result

        try:

            is_live, conf, is_error, msg = self.detect_action(keypoints, poses, boxes, action)

            result['alive'] = is_live
            result['conf'] = conf
            result['is_error'] = is_error
            result['msg'] = msg
        except Exception as e:
            result['is_error'] = True
            result['msg'] = f'detect_action error: {e}'
            logger.error(f'detect_action error: {e}')

        return result
    
    def detect_anti_face(self, images):

        image = images[0]
        _, _, box = self.face_2d_kpts(image)
        anti_score = 0
        real_face = False
        # anti_image = torch.FloatTensor(image).to(settings.TORCH_DEVICE)
        # anti_image = anti_image[:, :, (2, 1, 0)]
        # anti_image = anti_image.permute((2, 0, 1)).unsqueeze(0)
        # anti_image = anti_image.cpu().numpy()
        region_1 = self.get_anti_detect_region(image, box, 2.7)
        region_2 = self.get_anti_detect_region(image, box, 4)
        result_1 = face_anti_1.predict(region_1)
        result_2 = face_anti_2.predict(region_2)
        result = result_1 + result_2
        # logger.info('result:')
        # logger.info(result_1)
        # logger.info(result_2)
        # logger.info(result)
        anti_score, anti = torch.max(result, 1)
        anti_score = anti_score.item() / 2
        anti = anti.item()
        real_face = anti == 1 and anti_score > settings.ANTI_SCORE
        

        return real_face, anti_score
    
    def get_anti_detect_region(self, image, box, scale):
        """
        根据检测到的人脸框，按照指定scale倍数扩展人脸区域，并返回resize后的人脸区域
        :param src_image:       tensor -- [1, 3, h, w]
        :param box:             list -- [x_min, y_min, x_max, y_max]
        :param scale:           float -- 人脸区域扩展倍数， 2.7 for model_1, 4 for model_2
        :return:
                    tensor -- scaled face region that resize to (80, 80)
        """
        # src_height = src_image.shape[2]
        # src_width = src_image.shape[3]
        
        src_height = image.shape[0]
        src_width = image.shape[1]
        
        box_height = box[3] - box[1]
        box_width = box[2] - box[0]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        scale = min(src_height / box_height, src_width / box_width, scale)
        long_side = max(box_height, box_width)
        new_side = long_side * scale

        x_min = center_x - new_side / 2
        y_min = center_y - new_side / 2
        x_max = center_x + new_side / 2
        y_max = center_y + new_side / 2
        if x_min < 0:
            x_max -= x_min
            x_min = 0
        if y_min < 0:
            y_max -= y_min
            y_min = 0
        if x_max > src_width:
            x_min -= x_max - src_width
            x_max = src_width
        if y_max > src_height:
            y_min -= y_max - src_height
            y_max = src_height
        box = [
            [x_min, y_min], 
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max]
        ]

        region = self.crop_and_resize(image, box, (80, 80))

        return region


        
    def crop_and_resize(self, image, boxes, size) -> torch.Tensor:

        # logger.info(image.shape)

        # unpack input data
        dst_h = size[0]
        dst_w = size[1]

        points_src = np.array(boxes, dtype='float32')

        # points_dst = torch.Tensor([[
        #     [0, 0],
        #     [0, dst_w - 1],
        #     [dst_h - 1, 0],
        #     [dst_h - 1, dst_w - 1],
        # ]]).repeat(points_src.shape[0], 1, 1).numpy()

        points_dst = np.array(
            [
                [0, 0],
                [0, dst_w - 1],
                [dst_h - 1, 0],
                [dst_h - 1, dst_w - 1],
            ],
            dtype = 'float32'
        )
        # logger.info(f'points_dst: {points_dst.shape}\n{points_dst}')
        # logger.info(f'points_src: {points_src.shape}\n{points_src}')
        
        # compute transformation between points and warp
        dst_trans_src = cv2.getPerspectiveTransform(points_src, points_dst)


        
        # simulate broadcasting
        # logger.info(f'dst: {dst_trans_src.shape}')

        # dst_trans_src = torch.Tensor(dst_trans_src).expand(image.shape[0], -1, -1).cpu().numpy().astype(np.float32)

        # cv2.imwrite('pre_patches.jpg', image)
        # logger.info(f'dest_trans_src: {dst_trans_src.shape}\n{dst_trans_src}')
        patches = cv2.warpPerspective(image, dst_trans_src, (dst_h, dst_w))
        # cv2.imwrite('patches.jpg', patches)

        tr_patches = torch.FloatTensor(patches).to(settings.TORCH_DEVICE)
        tr_patches = tr_patches[:, :, (2, 1, 0)]
        tr_patches = tr_patches.permute((2, 0, 1)).unsqueeze(0)
        # logger.info(f'tr: {tr_patches.cpu().numpy().shape}')
        # anti_image = anti_image.cpu().numpy()


        return tr_patches

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[2], eye[6])
        B = dist.euclidean(eye[0], eye[4])
        ear = A / B
        return ear
    
    def eye_dist(self, eye):
        A = dist.euclidean(eye[2], eye[6])
        # B = dist.euclidean(eye[0], eye[4])
        return A
    def EB_dist(self, eye, brow):
        A = dist.euclidean(eye[2], brow)
        return A

    def mouth_aspect_ratio(self, mouth):
        # 计算两组垂直方向的欧式距离
        A = dist.euclidean(mouth[3], mouth[9])  # 51, 59
        B = dist.euclidean(mouth[14], mouth[18])  # 53, 57
        # 计算水平方向的距离
        C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
        D = dist.euclidean(mouth[12], mouth[16])
        mar = (A + B) / (C + D)
        return mar
    
    def mutationJudgment(self, sequence):
        mutation_detected = False
        mutation_degree = 0.0
        change_list = []

        for i in range(1, len(sequence)):
            previous_value = sequence[i - 1]
            current_value = sequence[i]

            if previous_value != 0:
                relative_change = abs(current_value - previous_value) / previous_value
            else:
                relative_change = abs(current_value - previous_value)

            if relative_change > 1:  # 这里的 0.5 是一个示例相对变化阈值，你可以根据具体情况调整
                mutation_detected = True
                mutation_degree += relative_change
            change_list.append(relative_change)
        # logger.info(f'change: {change_list}')
        plot(change_list, 'change')

        return mutation_detected, self.conf_function(mutation_degree - 1)

    def blink_detect(self, kpts):
        ear_list = []

        for kpt in kpts:
            leftEye = kpt[self.lStart:self.lEnd]
            rightEye = kpt[self.rStart:self.rEnd]


            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            EAR = (leftEAR+rightEAR)/2.0

            ear_list.append(EAR)
        # logger.info(ear_list)
        plot(ear_list, 'ear')
        

        return self.mutationJudgment(ear_list)
    
    
    def conf_function(self, x):
        return 1 - np.exp(-x)


    def mouth_detect(self, kpts):
        mouth = kpts[self.mStart:self.mEnd]

        mouthMAR = self.mouth_aspect_ratio(mouth)
        mar = mouthMAR

        if mar > self.MOUTH_AR_THRESH:
            return True
        # mouthHull = cv2.convexHull(mouth)
        return False
    
    def shake_detect(self, poses):
        left, right = 0, 0

        for po in poses:
            yaw = po[2]


            # logger.info(f'\n yaw: {yaw}')

            if yaw >= self.YAW_MAX:
                right += 1
            if yaw <= self.YAW_MIN:
                left += 1
            if right != 0 and left != 0:
                return True
        return False
    
    def nod_detect(self, poses):
        up = 0
        down = 0
        # pitch = po[0]

        for po in poses:
            pitch = po[0]

            # logger.info(f'pitch: {pitch}')

            if pitch>=self.PITCH_MAX:
                up+=1
            if pitch <= self.PITCH_MIN:
                down+=1
            if up !=0 and down!=0:
                return True
        
        return False



    def in_area(self, boxes):
        image_shape = self.img_shape
        box = boxes[0]

        image_center = (image_shape[0] / 2, image_shape[1] / 2)
        box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        distance = np.linalg.norm(np.array(image_center) - np.array(box_center))

        circle_radius = min(image_shape[0] / 2, image_shape[1] / 2)


        if distance < circle_radius:
            return True
        else:
            return False



    def predecte(self, images):
        keypoints = []
        poses = []
        boxes = []
        
        for i,frame in enumerate(images):
            try:
                kpts, po, box = self.face_2d_kpts(frame)
            except Exception as e:
                logger.error(f'face_2d_kpts error: {e}')
            if len(kpts)>90:
                keypoints.append(kpts)
                poses.append(po)
                boxes.append(box)
            else:
                logger.info(f'frame {i+1}th no face detected')
        
        return keypoints, poses, boxes

    def detect_action(self, keypoints, poses, boxes, action):
        is_live = False
        conf = 0
        is_error = False
        msg = None

        try:
            in_circle = self.in_area(boxes)
        except Exception as e:
            msg = f'in_area: {e}'

        if not in_circle:
            is_error = True
            msg = 'Face not inside the frame'

            return is_live, conf, is_error, msg
        
        if action == self.ACTIONS[1]:
            try:
                is_live, conf = self.blink_detect(keypoints)
            except Exception as e:
                msg = f'blink_detect error: {e}'
                is_error = True

        elif action == self.ACTIONS[2]:
            # gape
            for kpts in keypoints:
                is_live = self.mouth_detect(kpts)
                if is_live:
                    break

        elif action == self.ACTIONS[3]:
            # shake
            is_live = self.shake_detect(poses)

        elif action == self.ACTIONS[4]:
            # nod
            try:
                is_live = self.nod_detect(poses)
            except Exception as e:
                msg = f'nod_detect error: {e}'
                is_error = True

            

        return is_live, conf, is_error, msg

def test():
    alive_detector = OperateDetect()
    images=[]
    for i in range(1,30):
        images.append(cv2.imread(f'data/blink/image_{i}.jpg'))
    # frame = cv2.imread('./test.jpg')
    #status:[1,2,3,4] eyes,mouth,sheck,nod
    result = alive_detector(images,'nothing')
    logger.info(result)


if __name__ == '__main__':
    test()