
import cv2
import time


if __name__=='__main__':
    video_captur = cv2.VideoCapture(0)
    while True:
        ret, frame = video_captur.read()

        # cv2.imshow(frame)
        cv2.imshow('test.jpg', frame)
        # time.sleep(1)
        cv2.waitKey(100)