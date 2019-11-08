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

    print("image: {}; label: {}".format(images[0].shape, labels[0].shape))

    model = tf.keras.Sequential()
    dense_layer = tf.keras.layers.Dense(5, input_shape=images[0].shape)
    model.add(dense_layer)

    model.compile(loss="sparse_categorical_crossentropy",
        optimizer="SGD",
        metrics=['accuracy'])

    print(model.fit(images, labels, steps_per_epoch=10))
