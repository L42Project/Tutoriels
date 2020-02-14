import xmltodict
import numpy as np
import glob
import cv2

def xml_to_dataset(dir, size=None):
    tab_image=[]
    tab_label=[]
    for fichier in glob.glob(dir+"/*.xml"):
        with open(fichier) as fd:
            doc=xmltodict.parse(fd.read())
            image=doc['annotation']['filename']
            img=cv2.imread(dir+"/"+image)
            for obj in doc['annotation']['object']:
                xmin=int(obj['bndbox']['xmin'])
                xmax=int(obj['bndbox']['xmax'])
                ymin=int(obj['bndbox']['ymin'])
                ymax=int(obj['bndbox']['ymax'])
                if size is not None:
                    tab_image.append(cv2.resize(img[ymin:ymax, xmin:xmax], size))
                else:
                    tab_image.append(img[ymin:ymax, xmin:xmax])
                tab_label.append(obj['name'])

    l=list(set(tab_label))
    tab_one_hot=[]
    for e in tab_label:
        tab_one_hot.append(np.eye(len(l))[l.index(e)])
        
    return tab_image, l, tab_one_hot

tab_image, tab_label, tab_one_hot=xml_to_dataset("./", (32, 32))

for i in range(len(tab_image)):
    cv2.imshow('image', tab_image[i])
    print(tab_label[np.argmax(tab_one_hot[i])], tab_one_hot[i])
    cv2.waitKey()

