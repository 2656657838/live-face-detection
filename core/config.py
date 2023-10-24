from pydantic import BaseSettings
import torch


class Settings(BaseSettings):
    EYE_AR_THRESH: float
    EYE_MIN: float

    # CONSEC_FRAMES = 5  # 这个值被设置为 3，表明眼睛长宽比小于3时，接着三个连续的帧一定发生眨眼动作
    # COUNTER = 0  # 眼图长宽比小于EYE_AR_THRESH的连续帧的总数
    # TOTAL = 0  # 脚本运行时发生的眨眼的总次数

    MOUTH_AR_THRESH: float

    Nod_threshold: float
    shake_threshold: float
    
    
    PITCH_MAX: float
    PITCH_MIN: float
    YAW_MAX: float
    YAW_MIN: float

    ANTI_SCORE: float


    mStart: int
    mEnd: int
    nStart: int
    nEnd: int
    lStart: int
    lEnd: int
    rStart: int
    rEnd: int
    lBrow: int
    rBrow: int
    

    SERVER_HOST: str
    SERVER_PORT: int

    # GPU: str
    TORCH_DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


    class Config:
        env_prefix = 'FACE_'
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
