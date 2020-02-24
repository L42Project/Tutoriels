import tensorflow as tf
from tensorflow.keras import layers, models
import config

def block_resnet(input, filters, kernel_size, reduce=False):
    result=layers.Conv2D(filters, kernel_size, strides=1, padding='SAME')(input)
    result=layers.BatchNormalization()(result)
    result=layers.LeakyReLU(alpha=0.1)(result)

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
    result=layers.LeakyReLU(alpha=0.1)(result)
    result=layers.BatchNormalization()(result)
    return result

def model(nbr_classes, nbr_boxes, cellule_y, cellule_x):
    entree=layers.Input(shape=(config.largeur, config.hauteur, 3), dtype='float32')

    result=block_resnet(entree, 16, 3, False)
    result=block_resnet(result, 16, 3, True)

    result=block_resnet(result, 32, 3, False)
    result=block_resnet(result, 32, 3, True)

    result=block_resnet(result, 64, 3, False)
    result=block_resnet(result, 64, 3, False)
    result=block_resnet(result, 64, 3, True)

    result=block_resnet(result, 128, 3, False)
    result=block_resnet(result, 128, 3, False)
    result=block_resnet(result, 128, 3, True)

    result=layers.Conv2D(config.nbr_boxes*(5+config.nbr_classes), 1, padding='SAME')(result)
    sortie=layers.Reshape((config.cellule_y, config.cellule_x, config.nbr_boxes, 5+config.nbr_classes))(result)

    model=models.Model(inputs=entree, outputs=sortie)

    return model

