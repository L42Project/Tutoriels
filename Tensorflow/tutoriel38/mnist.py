import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import common

batch_size=64
nbr_cc=64 # Nombre de cartes de caracteristique du modele
embeddings_size=256 # Taille du vecteur embedding (sortie du réseau de neurone)

(x_train, y_train),(x_test, y_test)=tf.keras.datasets.mnist.load_data()

train_dataset=(x_train.reshape(-1, 28, 28, 1)/255).astype(np.float32)
test_dataset=(x_test.reshape(-1, 28, 28, 1)/255).astype(np.float32)

train_ds=tf.data.Dataset.from_tensor_slices((train_dataset, y_train)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((test_dataset, y_test)).batch(batch_size)

model=common.model_embedding(nbr_cc, embeddings_size)

common.write_labels_embs(model, test_ds, 'embeddings1.tsv', 'labels.tsv')

model.compile(
    optimizer=tf.keras.optimizers.Adam(1E-3),
    loss=tfa.losses.TripletSemiHardLoss())

model.fit(train_ds, epochs=5)

common.write_labels_embs(model, test_ds, 'embeddings2.tsv', None)
