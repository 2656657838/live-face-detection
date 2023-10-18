# numpy >= 1.20
import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

model_id = 'cv_mobilenet_face-2d-keypoints_alignment'
face_2d_keypoints = pipeline(Tasks.face_2d_keypoints, model=model_id, model_revision='v1.0.0')

# import ipdb
# ipdb.set_trace()



# output = face_2d_keypoints(img_path)

# the output contains point and pose
# print(output)
# for kpt in output['keypoints'][0]:
#     cv2.circle(image, (int(round(kpt[0])), int(round(kpt[1]))), radius=1, color=(0,0,255), thickness=-1)
# print(len(output['keypoints'][0]))
# cv2.imwrite('result.jpg', image)