import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
from sklearn import cluster
import sklearn

import common

batch_size=16
nbr_cc=64 # Nombre de cartes de caracteristique du modele
embeddings_size=256 # Taille du vecteur embedding (sortie du réseau de neurone)

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()

train_dataset=(x_train.reshape(-1, 28, 28, 1)/255).astype(np.float32)
test_dataset=(x_test.reshape(-1, 28, 28, 1)/255).astype(np.float32)

train_ds=tf.data.Dataset.from_tensor_slices((train_dataset, y_train)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((test_dataset, y_test)).batch(batch_size)

model=common.model_embedding(nbr_cc, embeddings_size)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1E-3),
    loss=tfa.losses.TripletSemiHardLoss())

model.fit(train_ds, epochs=5)

# Recherche des centroids des clusters
embeddings=model.predict(train_dataset)
kmeans=cluster.KMeans(n_clusters=len(set(y_train)))
kmeans.fit(embeddings)
centroids=kmeans.cluster_centers_

# Recherche du label des centroids
distances=sklearn.metrics.pairwise_distances(embeddings, centroids)
# shape distances: 6000, 10
# lmin contient le vecteur embeddings le plus proche de chacun des centroids
lmin=np.argmin(distances, axis=0)
labels_centroids=y_train[lmin]

# Calcul de précision de la base d'entrainement
result=np.equal(y_train, labels_centroids[np.argmin(distances, axis=-1)]).astype(np.float32)
print("Train: précision {:4.2%}".format(np.mean(result)))

# Calcul de précision de la base de test
embeddings=model.predict(test_dataset)
distances=sklearn.metrics.pairwise_distances(embeddings, centroids)
result=np.equal(y_test.astype(np.int32), labels_centroids[np.argmin(distances, axis=-1)]).astype(np.float32)
print("Test : précision {:4.2%}".format(np.mean(result)))
