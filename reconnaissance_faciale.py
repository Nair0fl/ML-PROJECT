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
    #Création du SVM
    clf = svm.SVC(kernel='linear')
    #Entrainement
    clf.fit(x_train, y_train)
    #Prédiction
    for key, x in enumerate(x_test):
        prediction = clf.predict([x])
        print("Prediction : ", prediction[0], " | Réalité : ", y_test[key])

if __name__ == "__main__":
    main()   