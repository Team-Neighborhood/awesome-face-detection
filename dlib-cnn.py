from __future__ import print_function
import numpy as np
import cv2
import dlib

cv2.namedWindow('show', 0)
cv2.imshow('show', np.zeros((5,5,3), dtype=np.uint8))
cv2.waitKey(500)

detector_hog = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')

bgr_img = cv2.imread('./test.jpg', 1)
print (bgr_img.shape)

### detection
list_time = []
for idx in range(10):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    start = cv2.getTickCount()
    (h, w) = bgr_img.shape[:2]

    rgb_img = cv2.resize(rgb_img, None, fx=0.5, fy=0.5)
    mmod_rects = detector_hog(rgb_img, 1)

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    list_time.append(time)
    # print ('elapsed time: %.3fms'%time)

### draw rectangle bbox
for i, mmod_rect in enumerate(mmod_rects):
    dlib_rect = mmod_rect.rect
    l = dlib_rect.left() * 2
    t = dlib_rect.top() * 2
    r = dlib_rect.right() * 2
    b = dlib_rect.bottom() * 2

    cv2.rectangle(bgr_img, (l,t), (r,b), (0,255,0), 2)

print ('average time: %.3f ms'%np.array(list_time[1:]).mean())

cv2.namedWindow('show', 0)
cv2.imshow('show', bgr_img)
cv2.waitKey()