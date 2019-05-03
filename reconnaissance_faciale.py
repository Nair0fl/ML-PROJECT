# -*- coding: utf-8 -*-
"""
Script basé en partie sur un de nos exercices de reconnaissance faciale
"""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn import svm
from timeit import default_timer as timer

class ReconnaissanceFaciale:
    """
    Classe regroupant différentes méthodes utiles pour tester la reconnaissance faciale
    d'une photo passée en entrée par rapport à un jeux de donnée fournit.
    """
    def test(self):
        """ Lance la reconnaissance faciale avec des données par défaut """
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
        for key, current in enumerate(x_test):
            self.reconnaitre_un_visage(clf, current, y_test[key])

    def entrainer(self, x_train, y_train, classifieur):
        """ Entrainement sur des photos """
        #Entrainement du classifieur
        timer_start = timer()
        classifieur.fit(x_train, y_train)
        timer_end = timer()
        #Calcul du nombre de photos dans le jeu d'entrainement
        nb_photos = len(x_train)
        #Calcul du nombre de personnes dans le jeu d'entrainement
        personnes = []
        for person in y_train:
            if person not in personnes:
                personnes.append(person)
        nb_personnes = len(personnes)
        #Affichage du nombre de photos et de personnes dans le jeu d'entrainement
        print("Je me suis entrainé sur", nb_photos, "photos représentant le visage de", nb_personnes, "personnes en", round(timer_end - timer_start, 6), "secondes")

    def reconnaitre_un_visage(self, classifieur, visage, reponse=None):
        """ Identifier un visage """
        #Tentative de prédiction
        prediction = classifieur.predict([visage])
        #Affichage de la prédiction
        if reponse != None:
            if prediction == reponse:
                print("J'ai correctement identifié", prediction[0])
            else:
                print("Je me suis trompé, je pensais qu'il s'agissait de", prediction[0], "mais c'était", reponse)
        else:
            print("Je pense qu'il s'agit de", prediction[0])
        return prediction[0]
