import tensorflow as tf
import numpy as np
import glob
import cv2
import model
import config

tab_images=[]
tab_labels=[]

def complete_dataset(files, value):
    for image in glob.glob(files):
        img=cv2.imread(image)
        img=cv2.resize(img, (config.size, config.size))
        tab_images.append(img)
        tab_labels.append([value])
        img=cv2.flip(img, 1)
        tab_images.append(img)
        tab_labels.append([value])
        img=cv2.flip(img, 0)
        tab_images.append(img)
        tab_labels.append([value])

complete_dataset(config.dir_pos+'\\*.png', 1.)
complete_dataset(config.dir_neg+'\\*.png', 0.)

tab_images=np.array(tab_images, dtype=np.float32)/255
tab_labels=np.array(tab_labels, dtype=np.float32)

index=np.random.permutation(len(tab_images))
tab_images=tab_images[index]
tab_labels=tab_labels[index]

#for i in range(len(tab_images)):
#    cv2.imshow('Camera', tab_images[i])
#    print("Label", tab_labels[i])
#    if cv2.waitKey()&0xFF==ord('q'):
#           quit()

model=model.model(config.size, 8)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(tab_images,
          tab_labels,
          validation_split=0.05,
          batch_size=64,
          epochs=30)
model.save('saved_model\\my_model')
