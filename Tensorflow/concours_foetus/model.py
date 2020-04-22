import tensorflow as tf
from tensorflow.keras import layers, models
import config

def block_resnet(input, filters, kernel_size, reduce, dropout=0.):
    result=layers.Conv2D(filters, kernel_size, strides=1, padding='SAME', activation='relu')(input)
    if dropout is not 0.:
        result=layers.Dropout(dropout)(result)
    if reduce is True:
        result=layers.Conv2D(filters, kernel_size, strides=2, padding='SAME')(result)
    else:
        result=layers.Conv2D(filters, kernel_size, strides=1, padding='SAME')(result)
        
    if input.shape[-1]==filters:
        if reduce is True:
            shortcut=layers.Conv2D(filters, 1, strides=2, padding='SAME')(input)
        else:
            shortcut=input
    else:
        if reduce is True:
            shortcut=layers.Conv2D(filters, 1, strides=2, padding='SAME')(input)
        else:
            shortcut=layers.Conv2D(filters, 1, strides=1, padding='SAME')(input)    
    result=layers.add([result, shortcut])
    if dropout is not 0.:
        result=layers.Dropout(dropout)(result)
    result=layers.Activation('relu')(result)
    result=layers.BatchNormalization()(result)
    return result

def model(nbr):
    entree=layers.Input(shape=(config.largeur, config.hauteur, 1), dtype='float32')

    result=block_resnet(entree, 2*nbr, 3, False, 0.3)
    result=block_resnet(result, 2*nbr, 3, False, 0.3)
    result=block_resnet(result, 2*nbr, 3, False, 0.3)
    result=block_resnet(result, 2*nbr, 3, True,  0.3)

    result=block_resnet(result, 4*nbr, 3, False, 0.4)
    result=block_resnet(result, 4*nbr, 3, False, 0.4)
    result=block_resnet(result, 4*nbr, 3, False, 0.4)
    result=block_resnet(result, 4*nbr, 3, False, 0.4)
    result=block_resnet(result, 4*nbr, 3, False, 0.4)
    result=block_resnet(result, 4*nbr, 3, False, 0.4)
    result=block_resnet(result, 4*nbr, 3, True,  0.4)

    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, False, 0.4)
    result=block_resnet(result, 8*nbr, 3, True,  0.4)

    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    result=block_resnet(result, 16*nbr, 3, False, 0.5)
    
    result=layers.AveragePooling2D()(result)    
    result=layers.Flatten()(result)
    sortie=layers.Dense(5, activation='sigmoid')(result)
    
    model=models.Model(inputs=entree, outputs=sortie)
    return model

