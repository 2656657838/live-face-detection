from loguru import logger

LOGGING_NAME = 'Face-Alive'

# 设置 Loguru 日志记录器
logger.remove()  # 移除默认的处理程序

# 添加控制台处理程序
logger.add("Face-Alive.log", level="INFO", format="{time} {level} {message}")
logger.add(lambda msg: print(msg, end='\n'), level='INFO')

LOGGER = logger.bind(name=LOGGING_NAME)

