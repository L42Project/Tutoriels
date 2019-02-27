# Tutoriel tensorflow

Ce tutoriel est visible dans la vidéo youtube qui se trouve à l'adresse suivante:
XXXXXXXXXXXXX

N'hésitez pas à faire des commentaires sur youtube

Pour utiliser ce script, vous devez récuperer les fichier MNIST sur le site suivant:
http://yann.lecun.com/exdb/mnist/
et les placer dans le repertoire ./mnist

Les graphiques suivants ont tous été générés avec un apprentissage sur 200 cycles. Cette apprentissage a pris entre 30 et 40 minutes sur un GeForce 1080

Fonction d'activation: sigmoid (tn.nn.sigmoid)
Temps approximatif de calcul sur une GeForce 1080: 35 minutes
![graph sigmoid](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_sigmoid.png)

Fonction d'activation: tangente hyperbolique (tn.nn.tanh)
![graph tangente hyperbolique](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_tanh.png)

Fonction d'activation: Unité de Rectification Linéaire (tn.nn.relu)
![graph Unité de Rectification Linéaire](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_relu.png)

Fonction d'activation: Leaky Relu (tn.nn.leaky_relu)
![graph Leaky Relu](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_leaku_relu.png)

Fonction d'activation: Unité Exponentielle Linéaire (tn.nn.selu)
![graph Unité Exponentielle Linéaire](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_selu.png)

Ci-dessous, l'ensemble des fonctions dans un même graphique:

Erreur sur la base d'entrainement
![graph fonctions d'activation](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_all.png)

Erreur sur la base de test:
![graph fonctions d'activation](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_all_2.png)

