# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:09:13 2019

@author: Clément
"""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn import svm

class ReconnaissanceFaciale:
    
    """ Lance la reconnaissance faciale avec des données par défaut """
    def test(self):
        #Chargement des données
        faces = fetch_olivetti_faces()
        #Séparation data / target
        data = faces['data']
        target = faces['target']
        #Création d'un jeu de train et de test
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42)
        #Création du SVM
        clf = svm.SVC(kernel='linear')
        #Entrainement
        self.entrainer(x_train, y_train, clf)
        #Prédiction
        for key, x in enumerate(x_test):
            self.reconnaitre_un_visage(clf, x, y_test[key])
    
    """ Entrainement sur des photos """
    def entrainer(self, x_train, y_train, classifieur):
        #Entrainement du classifieur
        classifieur.fit(x_train, y_train)
        #Calcul du nombre de photos dans le jeu d'entrainement
        nb_photos = len(x_train)
        #Calcul du nombre de personnes dans le jeu d'entrainement
        personnes = []
        for p in y_train:
            if p not in personnes:
                personnes.append(p)
        nb_personnes = len(personnes)
        #Affichage du nombre de photos et de personnes dans le jeu d'entrainement
        print("Je me suis entrainé sur", nb_photos, "photos représentant le visage de", nb_personnes,"personnes")
    
    """ Identifier un visage """
    def reconnaitre_un_visage(self, classifieur, visage, reponse=None):
        #Tentative de prédiction
        prediction = classifieur.predict([visage])
        #Affichage de la prédiction
        if reponse != None:
            if prediction == reponse:
                print("J'ai correctement identifié", prediction[0])
            else:
                print("Je me suis trompé, je pensais qu'il s'agissait de", prediction[0],"mais c'était", reponse)
        else:
            print("Je pense qu'il s'agit de", prediction[0])
        return prediction[0]