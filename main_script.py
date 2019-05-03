# -*- coding: utf-8 -*-
"""
Script permettant d'identifier les personnes souhaitées selon les données d'entrainement
"""
import cv2
import os
from sklearn import svm
from reconnaissance_faciale import ReconnaissanceFaciale
import numpy as np

def detect_face(img):
    """ Détecte les visages au sein de la photo passée en paramètre """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./lbpcascades/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

def draw_rectangle(img, rect):
    """ Dessine le rectangle autour de la tête des personnes """
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    """ Ajoute le nom des personnes identifiées au rectangle de leur tête """
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def detectAndDraw(test_img, rf, classifier):
    """ Identifie le visage sur la photo puis le met en valeur avec draw_rectangle et draw_text """
    img = test_img.copy()
    rects = detect_face(img)
    for rect in rects:
        (x, y, w, h) = rect
        face = img[y:y+h, x:x+w]
        draw_rectangle(img, rect)
        label = rf.reconnaitre_un_visage(classifier, face)
        draw_text(img, label, rect[0], rect[1]-5)

    return img

class dataset:
    def __init_():
        return

def loadTestDataset(path):
    """ Chargement les données de test """
    data = dataset()
    dirs = os.listdir(path)
    targets = []
    images = []
    labels=[]
    target=0
    for dir_name in dirs:
        path2 = path+"/"+dir_name+"/"
        imagePaths = [path2 + f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
        imagePaths = sorted(imagePaths)
        name = dir_name.split("_")[0]+" "+dir_name.split("_")[1]
        labels.append(name)
        for x in imagePaths:
            img=cv2.imread(x)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_face(img)
            for face in faces:
                (x, y, w, h) = face
                face = gray[y:y+h, x:x+w]
                targets.append(target)
                images.append(face)
        target+=1
    data.target = targets
    data.images = images
    data.labels=labels
    return data

def main():
    #Chargement des données d'entrainement
    faces = loadTestDataset("Data/training")
    y_train=[]
    i=0;
    for label in faces.target:
        y_train
    #Création du classifieur SVM
    classifier = svm.SVC(gamma=0.001)
    #print(faces.images)
    rf = ReconnaissanceFaciale()
    #Entrainement
    rf.entrainer(faces.images, faces.target, classifier)
    #Chargement des données de test
    entries = os.listdir('Data/test')
    #Pour chaque image à prédire on l'affiche et on tente de trouver le nom     des personnes sur la photo
    for entry in entries:
        test_img = cv2.imread("Data/test/"+entry)
        test_img = detectAndDraw(test_img, rf, classifier)
        cv2.imshow(entry, test_img)
    print("Prediction complete")
    
    #Fin du programme et fermeture des images
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()   