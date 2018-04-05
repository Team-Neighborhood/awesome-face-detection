from __future__ import print_function
import numpy as np
import cv2

cv2.namedWindow('show', 0)
cv2.imshow('show', np.zeros((5,5,3), dtype=np.uint8))
cv2.waitKey(500)

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')

bgr_img = cv2.imread('./test.jpg', 1)
print (bgr_img.shape)

### detection
list_time = []
for idx in range(10):
    start = cv2.getTickCount()
    (h, w) = bgr_img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    list_time.append(time)
    # print ('elapsed time: %.3fms'%time)

### draw rectangle bbox
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < 0.5:
        continue
    
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (l, t, r, b) = box.astype("int") # l t r b
    
    cv2.rectangle(bgr_img, (l, t), (r, b),
        (0, 255, 0), 2)
    
    text = "face: %.2f" % confidence
    text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = t #- 1 if t - 1 > 1 else t + 1
    cv2.rectangle(bgr_img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
    cv2.putText(bgr_img, text, (l, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

print ('average time: %.3f ms'%np.array(list_time[1:]).mean())

cv2.namedWindow('show', 0)
cv2.imshow('show', bgr_img)
cv2.waitKey()