# numpy >= 1.20
import sys
sys.path.append('.')
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from pathlib import Path
from typing import Any


class Face2dKeypointPredictor:

    def __init__(self) -> None:
        self.pipeline = pipeline(
            Tasks.face_2d_keypoints, 
            model=str(Path(__file__).parent / 'cv_mobilenet_face-2d-keypoints_alignment'), 
            model_revision='v1.0.0'
        )
        
    def postprocess(self, preds):

        if len(preds['keypoints']) == 0:
            return [],[],[]

        
        keypoints = preds['keypoints'][0]
        poses = preds['poses'][0]
        boxes = preds['boxes'][0]

        return keypoints, poses, boxes
    
    def __call__(self, img) -> Any:
        preds = self.pipeline(img)

        return self.postprocess(preds)