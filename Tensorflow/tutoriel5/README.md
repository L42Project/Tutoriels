# Tutoriel tensorflow

Ce tutoriel est visible dans la vidéo youtube qui se trouve à l'adresse suivante:
XXXXXXXXXXXXX

N'hésitez pas à faire des commentaires sur youtube

Les graphiques suivants montrent les différentes courbes d'apprentissage réalisées avec le programme du tutoriel #2. Seule la fonction d'activation est changée.
L'apprentissage se fait sur 200 cycles et prend environ 35 minutes avec une GeForce 1080; le temps reste le même quelque soit la fonction.

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

Erreur sur la base d entrainement

![graph base entrainement](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_all.png)

Erreur sur la base de test

![graph base de test](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_all_2.png)

