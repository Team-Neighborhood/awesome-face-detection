from __future__ import print_function
import face_recognition
import numpy as np
import cv2

cv2.namedWindow('show', 0)
cv2.imshow('show', np.zeros((5,5,3), dtype=np.uint8))
cv2.waitKey(500)

bgr_img = cv2.imread('./test.jpg', 1)
print (bgr_img.shape)

### detection
list_time = []
for idx in range(10):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    start = cv2.getTickCount()

    rgb_img = cv2.resize(rgb_img, None, fx=0.5, fy=0.5)
    bbs = face_recognition.face_locations(rgb_img, model='cnn')

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    list_time.append(time)
    # print ('elapsed time: %.3fms'%time)

### draw rectangle bbox
for bb in bbs:
    (t, r, b, l) = np.array(bb, dtype='int')*2
    cv2.rectangle(bgr_img, (l, t), (r, b), (0, 255, 0), 2)

print ('average time: %.3f ms'%np.array(list_time[1:]).mean())

cv2.namedWindow('show', 0)
cv2.imshow('show', bgr_img)
cv2.waitKey()