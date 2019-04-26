import cv2
from sklearn.model_selection import train_test_split
import os
from sklearn import svm
import scipy
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
    rects = detect_face(img)
    for rect in rects:
        draw_rectangle(img, rect)
        label=svr.predict(rect)
        draw_text(img, label, rect[0], rect[1]-5)
        
    return img

class dataset:
        def __init_():
                return
            
def loadTestDataset(path):
        data = dataset()
        dirs = os.listdir(path)
        targets = []
        filenames = []
        images = []
        for dir_name in dirs:
            path2=path+"/"+dir_name+"/"
            imagePaths = [ path2 + f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2,f))]
            imagePaths = sorted(imagePaths)
            name=dir_name.split("_")[0]+" "+dir_name.split("_")[1]

            for x in imagePaths:
                filename = os.path.basename(x)
                target = name
                target = target
                targets.append(target)
                filenames.append(filename)
                images.append(scipy.misc.imread(x,1))
    
        data.target = targets
        data.images =images
        data.filenames = filenames

        return data

def shape_data(data):
        n_samples = len(data)
        return data.reshape((n_samples, -1))
    
def train_classifer(data,target):
        classifier = svm.SVC(gamma=0.001)
        classifier.fit(data, target)
        return classifier



faces=loadTestDataset("Data/training")

# Training our classifer so it knows how to classify digits
print(faces.target)
x_train, X_test, y_train, y_test = train_test_split(faces.images, faces.target,
                                                        random_state=42,
                                                        test_size=0.1)
classifier = svm.SVC(gamma=0.001)
print(y_test)
print(x_train)

classifier.fit(test_x, test_x)
#load test images


print("Prediction complete")

#display both images
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
