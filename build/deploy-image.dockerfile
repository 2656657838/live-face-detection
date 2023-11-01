FROM bywin.harbor.com:52/nvidia/pytorch-22.07-py3-ultralytics-8.0.117-pycuda-2022.1:v0.0.1
WORKDIR /workspace


RUN pip install loguru -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pydantic[dotenv] -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install addict -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install yapf -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install gast -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install simplejson -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install oss2 -i https://pypi.tuna.tsinghua.edu.cn/simple