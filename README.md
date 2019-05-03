
# ML-PROJECT

Projet d'initiation à la reconnaissance faciale réalisé dans le cadre du cours de Machine Learning Ynov 2019

## Contributeurs:
* [Antoine Drouard](https://github.com/Coblestone)
* [Benoît Cochet](https://github.com/BenoitCochet)
* [Clément Caillaud](https://github.com/ClementCaillaud)
* [Florian Boche](https://github.com/Nair0fl)

## Avancement

 - [X] Construire un jeu de données (entrainement et test)
 - [X] Faire un algorithme de reconnaissance
 - [X] Identifier et délimiter les visages sur une image
 - [ ] Reconnaître un visage depuis la webcam

 Rendu du projet le vendredi 03 mai 2019

## Installation

Ce projet nécessite l'installation des éléments suivants :
 - Python 3.7.1
 - sklearn
 - cv2
 - os

## Utilisation

**Préparation des données**

Dans le dossier *Data/training/*, placer un dossier par personne, contenant les images du visage de cette personne qui serviront à l'apprentissage. Le nom du dossier doit être de la forme *prenom_nom*

Dans le dossier *Data/test/*, placer les images servant à la prédiction. L'algorithme tentera de trouver à qui appartient chaque visage

Afin d'accélérer l'exécution du programme, il est recommandé de redimensionner les images

**Exécution de l'algorithme**

Exécuter le fichier *main_script.py*
Après une phase d'apprentissage, le programme va afficher les photos présentes dans *Data/test/*, encadrer les visages, et indiquer à coté le nom de la personne qu'il pense avoir identifié
Appuyer sur une touche pour arrêter l'exécution
