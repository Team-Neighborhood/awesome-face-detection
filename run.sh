#!/bin/bash

SHOW=False

python ocv-dnn.py --with_draw $SHOW
python ocv-haar.py --with_draw $SHOW
python dlib-hog.py --with_draw $SHOW
python dlib-cnn.py --with_draw $SHOW
python fr-hog.py --with_draw $SHOW
python fr-cnn.py --with_draw $SHOW
python dan-mtcnn.py --with_draw $SHOW
python yxl-s3fd.py --with_draw $SHOW
python di-insight.py --with_draw $SHOW