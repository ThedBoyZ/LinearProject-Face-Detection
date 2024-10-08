from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
def detect_face(img, detector):
    # detect faces in the image
    faces = detector.detect_faces(img)
    return faces

def draw_faces(img, faces):
    # draw bounding box for each face
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return img

# cap = cv2.VideoCapture(0)
detector = MTCNN()

import os
path = 'augmented'
files = os.listdir(path)
images = []
# plt.figure(figsize=(10, 10))
for i,file in enumerate(files):
    img = cv2.imread(path + '/' + file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detect_face(img, detector)
    img = draw_faces(img, faces)
    images.append(img)
    plt.subplot(330 + 1 + i)
    plt.imshow(img)
    plt.axis('off')
    plt.title(file)
    plt.show()
    cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # break