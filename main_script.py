import cv2
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVC

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

def detectAndDraw(test_img,svr):
    img = test_img.copy()
    rects = detect_face(img).shape
    for rect in rects:
        draw_rectangle(img, rects)
        label=svr.predict(rects)
        draw_text(img, label, rects[0], rects[1]-5)
        
    return img

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    names=[]
    faces=[]
    for dir_name in dirs:
        if dir_name != "test":
            img_folder_path=data_folder_path+"/"+dir_name
            img_folder = os.listdir(img_folder_path)
            name=dir_name.split("_")[0]+" "+dir_name.split("_")[1]
            for imgs in img_folder:
                imgs_path=img_folder_path+"/"+imgs
                img_path=img_folder_path+"/"+imgs
                test_img = cv2.imread(img_path)
                faces.append(detect_face(test_img).shape)
                names.append(name)
    return faces,names;

x,y=prepare_training_data("Data/training")
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
svr = SVC(gamma='scale')
svr.fit(x,y)
#load test images
entries = os.listdir('Data/test')
for entry in entries:
    test_img = cv2.imread("Data/test/"+entry)
    test_img = detectAndDraw(test_img,svr)
    cv2.imshow(entry, test_img)
    

print("Prediction complete")

#display both images
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
