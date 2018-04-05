# from PIL import ImageDraw
import cv2


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    # img_copy = img.copy()
    # draw = ImageDraw.Draw(img_copy)
    draw = img.copy()

    for b in bounding_boxes:
        b = [int(round(value)) for value in b]
        # print (b)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
        

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(draw, (p[i] , p[i + 5]), 1, (255,0,0), -1)
        
    return draw
