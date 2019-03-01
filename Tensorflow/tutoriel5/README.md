# Tutoriel tensorflow

Ce tutoriel est visible dans la vidéo youtube qui se trouve à l'adresse suivante:
https://www.youtube.com/watch?v=QaazrkwooFA

N'hésitez pas à faire des commentaires sur youtube

## Test de différentes fonctions sur la programme du tutoriel #2

Les graphiques suivants montrent les différentes courbes d'apprentissage réalisées avec le programme du tutoriel #2. Seule la fonction d'activation est changée.
L'apprentissage se fait sur 200 cycles et prend environ 35 minutes avec une GeForce 1080; le temps reste le même quelque soit la fonction.

#### Fonction d'activation: sigmoid (tn.nn.sigmoid)
![graph sigmoid](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/images/Figure_sigmoid.png)

#### Fonction d'activation: tangente hyperbolique (tn.nn.tanh)
![graph tangente hyperbolique](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/images/Figure_tanh.png)

#### Fonction d'activation: Unité de Rectification Linéaire (tn.nn.relu)
![graph Unité de Rectification Linéaire](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/images/Figure_relu.png)

#### Fonction d'activation: Leaky Relu (tn.nn.leaky_relu)
![graph Leaky Relu](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/images/Figure_leaku_relu.png)

#### Fonction d'activation: Unité Exponentielle Linéaire (tn.nn.selu)
![graph Unité Exponentielle Linéaire](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/images/Figure_selu.png)

#### Ci-dessous, l'ensemble des fonctions dans un même graphique:

#### Erreur sur la base d entrainement:
![graph base entrainement](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/images/Figure_all.png)

#### Erreur sur la base de test:
![graph base de test](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/images/Figure_all_2.png)

## VGGNet sur CIFAR10

N'oubliez pas de récuperer la base cifar10 (binary version) à l'adresse suivante:
https://www.cs.toronto.edu/~kriz/cifar.html

L'entrainement prend environ 2h40 sur une GeForce 1080

#### Graph de l'entrainement:
![graph entrainement](https://github.com/L42Project/Tutoriels/blob/master/Tensorflow/tutoriel5/Figure_cifar_vgg.png)

