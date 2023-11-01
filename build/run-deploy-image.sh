# docker run -it --name face-deploy-tm -p 12728:80 -v $(pwd):/workspace bywin.harbor.com:52/nvidia/pytorch-22.07-py3-ultralytics-8.0.117-pycuda-2022.1:v0.0.1 /bin/bash 
docker run -it --name face-test -p 12728:80 -v $(pwd):/workspace python_backend_face:v2 /bin/bash
