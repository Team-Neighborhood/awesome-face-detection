from __future__ import print_function
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()

detector_haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

bgr_img = cv2.imread('./test.jpg', 1)
print (bgr_img.shape)

### detection
list_time = []
for idx in range(10):
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    start = cv2.getTickCount()
    (h, w) = bgr_img.shape[:2]

    gray_img = cv2.resize(gray_img, None, fx=0.5, fy=0.5)
    bbs = detector_haar.detectMultiScale(gray_img, 1.1)

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    list_time.append(time)
    # print ('elapsed time: %.3fms'%time)

print ('ocv-haar average time: %.3f ms'%np.array(list_time[1:]).mean())

### draw rectangle bbox
if args.with_draw == 'True':
    for bb in bbs:
        (l, t, w, h) = bb*2
        cv2.rectangle(bgr_img, (l, t), (l+w, t+h), (0, 255, 0), 2)

    cv2.namedWindow('show', 0)
    cv2.imshow('show', bgr_img)
    cv2.waitKey()