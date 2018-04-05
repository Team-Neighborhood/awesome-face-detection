
from __future__ import print_function
from mtcnn import detect_faces, show_bboxes
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()

bgr_img = cv2.imread('test.jpg', 1)
print (bgr_img.shape)

### detection
list_time = []
for idx in range(10):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    start = cv2.getTickCount()
    bounding_boxes, landmarks = detect_faces(rgb_img)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    list_time.append(time)

print ('mtcnn average time: %.3f ms'%np.array(list_time[1:]).mean())

### draw rectangle bbox
if args.with_draw == 'True':
    for b in bounding_boxes:
        b = [int(round(value)) for value in b]
        cv2.rectangle(bgr_img, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
        
    for p in landmarks:
        for i in range(5):
            cv2.circle(bgr_img, (p[i] , p[i + 5]), 3, (255,0,0), -1)

    cv2.namedWindow('show', 0)
    cv2.imshow('show', bgr_img)
    cv2.waitKey()
