import cv2

import os

import numpy as np

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./lbpcascades/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    return faces

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def detectAndDraw(test_img):
    img = test_img.copy()
    rects = detect_face(img)
    for rect in rects:
        (x, y, w, h)=rect
        draw_rectangle(img, rect)
        draw_text(img, "", rect[0], rect[1]-5)
    
    return img


#load test images
test_img2 = cv2.imread("Data/antoinedrouard.png")
test_img1 = cv2.imread("Data/clementcaillaud.png")

test_img1 = detectAndDraw(test_img1)
test_img2 = detectAndDraw(test_img2)

print("Prediction complete")

#display both images
cv2.imshow("Image1", test_img2)
cv2.imshow("Image2", test_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
