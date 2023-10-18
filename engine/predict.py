from typing import Any
from engine.modelscope_pipeline import face_2d_keypoints

class Face2dKeypointPredictor:

    def __init__(self) -> None:
        self.pipeline = face_2d_keypoints
        
    def postprocess(self, preds):

        
        keypoints = preds['keypoints'][0]

        return keypoints
    
    def __call__(self, img) -> Any:
        preds = self.pipeline(img)
        keypoints = self.postprocess(preds)

        return keypoints