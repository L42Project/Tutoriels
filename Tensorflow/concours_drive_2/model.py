import tensorflow as tf
from tensorflow.keras import layers, models

def LossDice(y_true, y_pred):
  numerateur  =tf.reduce_sum(y_true*y_pred, axis=(1, 2))
  denominateur=tf.reduce_sum(y_true+y_pred, axis=(1, 2))
  dice=2*numerateur/(denominateur+1E-4)
  return 1-dice
  
def LossJaccard(y_true, y_pred):
  intersection=tf.reduce_sum(y_true*y_pred, axis=(1, 2))
  union       =tf.reduce_sum(y_true+y_pred, axis=(1, 2))
  jaccard=intersection/(union-intersection+1E-4)
  return 1-jaccard

def model(nbr):
    entree=layers.Input(shape=(576, 560, 3), dtype='float32')

    result=layers.Conv2D(nbr, 3, activation='relu', padding='same')(entree)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result1=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result1)

    result=layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result2=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result2)

    result=layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result3=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result3)

    result=layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result4=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result4)

    result=layers.Conv2D(8*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result4], axis=3)

    result=layers.Conv2D(8*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result3], axis=3)
    
    result=layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result2], axis=3)
    
    result=layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    
    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result1], axis=3)
    
    result=layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    sortie=layers.Conv2D(1, 1, activation='sigmoid', padding='same')(result)

    model=models.Model(inputs=entree, outputs=sortie)
    return model
