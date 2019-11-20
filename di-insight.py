#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse

import cv2
import time
import numpy as np
from PIL import Image

import _add_path
import sys

import insightface

parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--model', type=str,
                    default='S3FD/weights/s3fd.pth', help='trained model')
parser.add_argument('--thresh', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--with_draw', default='True')
                    
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def detect_image(net, img_orig, thresh, scale=1/2):
    image = img_orig 
    height, width, _ = image.shape
    
    image = cv2.resize(image, None, fx=scale, fy=scale)

    detections, landmark = net.detect(image, threshold=thresh, scale=1.0)
    
    img = img_orig.copy()

    list_bbox_ltrb = []
    for i in range(len(detections)):
        if detections[i][-1] > thresh:
            bbox_ltrb = detections[i][:4] * (1/scale)
            conf = detections[i][-1]
            list_bbox_ltrb.append(bbox_ltrb.astype(np.int))

    return list_bbox_ltrb

if __name__ == '__main__':
    net = insightface.model_zoo.get_model('retinaface_r50_v1')
    net.prepare(ctx_id=0, nms=0.6)

    bgr_img = cv2.imread('./test.jpg', 1)
    print (bgr_img.shape)

    ### detection
    list_time = []
    for idx in range(10):
        start = cv2.getTickCount()
        (h, w) = bgr_img.shape[:2]

        list_bbox_ltrb = detect_image(net, bgr_img, args.thresh)

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        list_time.append(time)
        # print ('elapsed time: %.3fms'%time)

    print ('insightface average time: %.3f ms'%np.array(list_time[1:]).mean())

    ### draw rectangle bbox
    if args.with_draw == 'True':
        for bb in list_bbox_ltrb:
            (l, t, r, b) = bb
            cv2.rectangle(bgr_img, (l, t), (r, b), (0, 255, 0), 2)

        cv2.namedWindow('show', 0)
        cv2.imshow('show', bgr_img)
        cv2.waitKey()