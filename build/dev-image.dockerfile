FROM bywin.harbor.com:52/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.1
WORKDIR /workspace

RUN pip install imutils -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install loguru -i https://pypi.tuna.tsinghua.edu.cn/simple
