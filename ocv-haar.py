import numpy as np
import cv2

cv2.namedWindow('show', 0)
cv2.imshow('show', np.zeros((5,5,3), dtype=np.uint8))
cv2.waitKey(500)

detector_haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

bgr_img = cv2.imread('./test.jpg', 1)
print (bgr_img.shape)

### detection
list_time = []
for idx in range(10):
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    start = cv2.getTickCount()
    (h, w) = bgr_img.shape[:2]

    bbs = detector_haar.detectMultiScale(gray_img, 1.1)

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    list_time.append(time)
    # print ('elapsed time: %.3fms'%time)

### draw rectangle bbox
for bb in bbs:
    (l, t, w, h) = bb
    cv2.rectangle(bgr_img, (l, t), (l+w, t+h), (0, 255, 0), 2)

print ('average time: %.3f ms'%np.array(list_time[1:]).mean())

cv2.namedWindow('show', 0)
cv2.imshow('show', bgr_img)
cv2.waitKey()