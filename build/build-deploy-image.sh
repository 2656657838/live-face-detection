docker build . \
-f build/deploy-image.dockerfile \
-t bywin.harbor.com:52/banyun/live-face-detection:$1-deploy 
--platform linux/amd64 
# && docker push bywin.harbor.com:52/banyun/algorithm-object-detection:$1-dev
