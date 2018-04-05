# Awesome face detection

Compare face detectors - Dlib, OpenCV, Others..
<br>
<br>
<p align="center">
    <img src='./result.png' width=70%>
    <br>
    We are neighborhood
</p>

---

<br>

## Processing time

Test image : HD (720p)

Test on **Intel i7-6700K & GTX1080**.

| ocv-dnn | ocv-haar | dlib-hog | dlib-cnn | fr-hog | fr-cnn | mtcnn |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 17.79ms | 42.31ms | 108.61ms | 42.17ms | 108.50ms | 39.91ms | 334.38ms |

<br>

Test on **MacBook pro retina 2014 mid**.

| ocv-dnn | ocv-haar | dlib-hog | dlib-cnn | fr-hog | fr-cnn | mtcnn |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 46.53ms | 88.47ms | 174.81ms | 3276.62ms | 174.63ms | 3645.53ms | 928.752ms |

<br>

## Requirements

- Python 3.6
- OpenCV 3.4.0 (option: build from src with highgui)
- Dlib 19.10.0
- face_recognition 1.2.1
- pytorch 0.3.1

## Usage  

First, install libs

    pip install opencv-contrib-python
    pip install torch
    pip install dlib
    pip install face_recognition

Second, check run-time for each algorithm.

    ./run.sh

Of course, You can execute each file. and watch the result image (need opencv high gui)

    python dlib-hog.py

## Now, Select face detector you need!

<br><br>

---

## Algorithm

opencv haar cascade
 - https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html

opencv caffe based dnn (res-ssd)
 - https://github.com/opencv/opencv/tree/master/samples/dnn

dlib hog
 - http://dlib.net/

dlib cnn
 - http://blog.dlib.net/2016/10/easily-create-high-quality-object.html

face-recognition (dlib-based)
 - https://github.com/ageitgey/face_recognition

mtcnn
 - https://github.com/TropComplique/mtcnn-pytorch (code)
 - https://arxiv.org/abs/1604.02878 (paper)