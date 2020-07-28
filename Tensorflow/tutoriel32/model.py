from tensorflow.keras import layers, models

# Fonction d'activation à tester: sigmoid, tanh, relu, 

def model(nbr_sortie, nbr_cc):
    entree=layers.Input(shape=(75, 100, 3), dtype='float32')

    result=layers.Conv2D(nbr_cc, 5, activation='relu', padding='same')(entree)
    result=layers.Conv2D(nbr_cc, 5, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.MaxPool2D()(result)

    result=layers.Conv2D(2*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.Conv2D(2*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(2*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.Conv2D(2*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.MaxPool2D()(result)

    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.Conv2D(4*nbr_cc, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.MaxPool2D()(result)
    
    result=layers.Flatten()(result)
    result=layers.Dense(1024, activation='relu')(result)
    result=layers.Dense(1024, activation='relu')(result)
    result=layers.BatchNormalization()(result)
    sortie=layers.Dense(nbr_sortie, activation='softmax')(result)

    model=models.Model(inputs=entree, outputs=sortie)
    return model

