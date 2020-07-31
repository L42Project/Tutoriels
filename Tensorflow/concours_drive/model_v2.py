import tensorflow as tf
from tensorflow.keras import layers, models

def dice_loss(y_true, y_pred):
  numerator=2*tf.reduce_sum(y_true*y_pred, axis=(1, 2))
  denominator=tf.reduce_sum(y_true+y_pred, axis=(1, 2))
  dice=numerator/denominator
  return 1-dice

def jaccard_loss(y_true, y_pred):
  intersection=tf.reduce_sum(y_true*y_pred, axis=(1, 2))
  union=tf.reduce_sum(y_true+y_pred, axis=(1, 2))
  jaccard=intersection/(union-intersection)
  return 1-jaccard

def tversky_loss(y_true, y_pred):
  beta=0.5
  numerator=tf.reduce_sum(y_true*y_pred, axis=-1)
  denominator=y_true*y_pred+beta*(1-y_true)*y_pred+(1-beta)*y_true*(1-y_pred)
  return 1-(numerator+1)/(tf.reduce_sum(denominator, axis=-1)+1)

def model():
    entree=layers.Input(shape=(576, 560, 3), dtype='float32')

    result=layers.Conv2D(64, 3, activation='relu', padding='same')(entree)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)

    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result1=layers.Dropout(0.5)(result)

    result=layers.MaxPool2D()(result1)

    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result2=layers.Dropout(0.5)(result)

    result=layers.MaxPool2D()(result2)

    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result3=layers.Dropout(0.5)(result)

    result=layers.MaxPool2D()(result3)

    result=layers.Conv2D(512, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(512, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result4=layers.Dropout(0.5)(result)

    result=layers.MaxPool2D()(result4)

    result=layers.Conv2D(1024, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(512, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result4], axis=3)

    result=layers.Conv2D(512, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result3], axis=3)
    
    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result2], axis=3)
    
    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    
    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result1], axis=3)
    
    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)
    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Dropout(0.5)(result)

    sortie=layers.Conv2D(1, 1, activation='sigmoid', padding='same')(result)

    model=models.Model(inputs=entree, outputs=sortie)
    return model

def model2():
    entree=layers.Input(shape=(576, 560, 3), dtype='float32')

    result=layers.Conv2D(32, 3, activation='relu', padding='same')(entree)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(32, 3, activation='relu', padding='same')(result)
    result1=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result1)

    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result2=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result2)

    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result3=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result3)

    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result4=layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result4)

    result=layers.Conv2D(512, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result4], axis=3)

    result=layers.Conv2D(256, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result3], axis=3)
    
    result=layers.Conv2D(128, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result2], axis=3)
    
    result=layers.Conv2D(64, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(32, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    
    result=layers.UpSampling2D()(result)
    result=tf.concat([result, result1], axis=3)
    
    result=layers.Conv2D(32, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)
    result=layers.Conv2D(32, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    sortie=layers.Conv2D(1, 1, activation='sigmoid', padding='same')(result)

    model=models.Model(inputs=entree, outputs=sortie)
    return model

def model3(nbr):
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
