import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf
import io

def write_labels_embs(model, ds, file_embeddings, file_labels):
    embeddings=model.predict(ds)
    np.savetxt(file_embeddings, embeddings, delimiter='\t')
    if file_labels is not None:
        fichier=io.open(file_labels, 'w', encoding='utf-8')
        for images, labels in ds:
            [fichier.write("{:d}\n".format(x)) for x in labels]
        fichier.close()

def model_embedding(nbr_cc, embeddings_size):
    entree=layers.Input(shape=(28, 28, 1), dtype=tf.float32)

    result=layers.Conv2D(nbr_cc, 3, activation='relu', padding='same')(entree)
    result=layers.MaxPool2D()(result)
    result=layers.Conv2D(nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.MaxPool2D()(result)
    result=layers.Flatten()(result)
    result=layers.Dense(embeddings_size, activation=None)(result)
    embeddings=layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(result)

    model=models.Model(inputs=entree, outputs=embeddings)
    return model
