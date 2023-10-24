# from config import config
from flask import Flask, request
from core.logger import LOGGER as logger
from core.config import settings
signal_app = Flask(__name__)
signal_app.logger = logger
import base64
import numpy as np
import cv2
from alg import aliveDetector

host = settings.SERVER_HOST
port = settings.SERVER_PORT



message = {
    "images": None,
    "action": None
    }

def str2np(images):
    data = []
    for i,img_str in enumerate(images):
        img_byte = base64.b64decode(img_str)
        nparr = np.frombuffer(img_byte, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(f'data/test/test_{i}.jpg', image)
        data.append(image)
    return data




@signal_app.route('/face-alive-detect', methods=['POST'])
def detect():
    global images_data
    global is_get
    message = request.json
    action = message['action']

    result = {
            "action": action,
            "alive":False,
            "conf":0,
            "is_error":True,
            "msg": None
        }

    try:
        images = str2np(message['images'])
        logger.info(f'got {len(images)} images')
    except Exception as e:
        result['msg'] =  'Image encoding error'
        return result

    

    if action not in aliveDetector.ACTIONS:
        result["msg"] = 'Action not within detection range or spelling error'
        return result

    try:
        result = aliveDetector(images,action)
    except Exception as e:
        result['msg'] = e
        logger.error(e)

    logger.info(result)
    
    return result

def run_app():
    logger.info(f'创建接口: http://{host}:{port}/face-alive-detect')
    try:
        signal_app.run(host=host, port=port)
    except Exception as e:
        logger.error(f'创建接口失败:{e}')

if __name__=='__main__':
    run_app()


