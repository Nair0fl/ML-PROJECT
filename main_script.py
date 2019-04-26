import cv2
from sklearn.model_selection import train_test_split
import os
from sklearn import svm
import scipy
import numpy as np
from reconnaissance_faciale import ReconnaissanceFaciale 

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

def detectAndDraw(test_img,rf,classifier):
    img = test_img.copy()
    rects = detect_face(img)
    for rect in rects:
        draw_rectangle(img,rect )
        label=rf.reconnaitre_un_visage(classifier,rect)
        draw_text(img, label, rect[0], rect[1]-5)
        
    return img

class dataset:
        def __init_():
                return
            
def loadTestDataset(path):
        data = dataset()
        dirs = os.listdir(path)
        targets = []
        images = []
        for dir_name in dirs:
            path2=path+"/"+dir_name+"/"
            imagePaths = [ path2 + f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2,f))]
            imagePaths = sorted(imagePaths)
            name=dir_name.split("_")[0]+" "+dir_name.split("_")[1]

            for x in imagePaths:
                faces=detect_face(cv2.imread(x))
                target = name
                for face in faces:
                    targets.append(target)
                    images.append(face)
    
        data.target = targets
        data.images =images

        return data


faces=loadTestDataset("Data/training")
classifier = svm.SVC(gamma=0.001)
rf= ReconnaissanceFaciale()
rf.entrainer(faces.images,faces.target,classifier)
entries = os.listdir('Data/test')

for entry in entries:
    test_img = cv2.imread("Data/test/"+entry)
    test_img = detectAndDraw(test_img,rf,classifier)
    cv2.imshow(entry, test_img)

print("Prediction complete")
#load test images


#display both images
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
