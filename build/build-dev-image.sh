docker build . \
-f build/dev-image.dockerfile \
-t bywin.harbor.com:52/banyun/live-face-detection:$1-dev 
--platform linux/amd64 
# && docker push bywin.harbor.com:52/banyun/algorithm-object-detection:$1-dev
