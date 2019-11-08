import tensorflow as tf

#from loadData import *
#from setupNN import *

import loadData
import setupNN

if __name__ == '__main__':

    images = []
    labels = []
    for file in loadData.load_images():
        img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
        images.append(img)
        labels.append(label)

    for i in range(0, len(images)):
        print("{} {}".format(images[i].shape, labels[i]))