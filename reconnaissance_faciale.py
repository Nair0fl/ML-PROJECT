# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:09:13 2019

@author: Clément
"""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn import svm

def main():
    #Chargement des données
    faces = fetch_olivetti_faces()
    #Séparation data / target
    data = faces['data']
    target = faces['target']
    #Création d'un jeu de train et de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)
    
    clf = svm.SVC(kernel='linear')
    clf = entrainement(clf, x_train, y_train)
    prediction(clf, x_test, y_test)
    
def entrainement(classifieur, x_train, y_train):
    #Entrainement
    classifieur.fit(x_train, y_train)
    return classifieur

def prediction(classifieur, x_test, y_test):
    #Tests de prédicition sur le jeu de test
    for key, x in enumerate(x_test):
        prediction = classifieur.predict([x])
        print("Prediction : ", prediction[0], " | Réalité : ", y_test[key])

if __name__ == "__main__":
    main()   