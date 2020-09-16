from tensorflow.keras import layers, models

def generator_model():
    entree_bruit =layers.Input(shape=(100), dtype='float32')
    entree_classe=layers.Input(shape=(10),  dtype='float32')

    result=layers.concatenate([entree_bruit, entree_classe])

    result=layers.Dense(7*7*256, use_bias=False)(result)
    result=layers.BatchNormalization()(result)
    result=layers.LeakyReLU()(result)

    result=layers.Reshape((7, 7, 256))(result)

    result=layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(result)
    result=layers.BatchNormalization()(result)
    result=layers.LeakyReLU()(result)

    result=layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(result)
    result=layers.BatchNormalization()(result)
    result=layers.LeakyReLU()(result)

    sortie=layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(result)

    model=models.Model(inputs=[entree_bruit, entree_classe], outputs=sortie)
    return model

def discriminator_model():
    entree_image =layers.Input(shape=(28, 28, 1), dtype='float32')
    entree_classe=layers.Input(shape=(10),        dtype='float32')

    result1=layers.Dense(28*28, use_bias=False)(entree_classe)
    result1=layers.Reshape((28, 28, 1))(result1)

    result=layers.concatenate([entree_image, result1])

    result=layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(result)
    result=layers.LeakyReLU()(result)
    result=layers.Dropout(0.3)(result)

    result=layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(result)
    result=layers.LeakyReLU()(result)
    result=layers.Dropout(0.3)(result)

    result=layers.Flatten()(result)
    sortie=layers.Dense(1)(result)

    model=models.Model(inputs=[entree_image, entree_classe], outputs=sortie)
    return model
