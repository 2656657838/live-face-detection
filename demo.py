from alg import aliveDetector
import cv2

if __name__ == '__main__':
    images=[]
    for i in range(1,30):
        images.append(cv2.imread(f'data/blink/image_{i}.jpg'))
    
    result = aliveDetector(images, 'nothing')
    print(result)