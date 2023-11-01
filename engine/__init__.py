from .face_anti import FaceAntiM1, FaceAntiM2
from .Face2DKeypointsPredictor import Face2dKeypointPredictor
from core.logger import LOGGER as logger


face_anti_1 = FaceAntiM1()
face_anti_2 = FaceAntiM2()
logger.info('静默活体检测模块加载完成')

face_2d_kpts = Face2dKeypointPredictor()
logger.info('关键点检测模块加载完成')
__all__ = ['face_anti_1', 'face_anti_2', 'face_2d_kpts']