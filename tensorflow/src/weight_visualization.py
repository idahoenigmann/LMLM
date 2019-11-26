import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from collections import defaultdict
import loadData

def plot_conv_weights(model, layer):
    W = model.get_layer(name=layer).get_weights()[0]

    if len(W.shape) == 2:
        W = W.reshape((4 * 18, 64))
        plt.imshow(W)

    if len(W.shape) == 4:
        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3]))
        fig, axs = plt.subplots(5, 5, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=.001)
        axs = axs.ravel()
        for i in range(25):
            axs[i].imshow(W[:, :, i])
            axs[i].set_title(str(i))
    plt.show()


if __name__ == '__main__':
    file = loadData.load_images()[0]
    image_shape = loadData.get_image_dimensions(str(file)[0:(str(file).rfind("."))])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 5, activation=tf.keras.activations.relu, input_shape=(image_shape[0], image_shape[1],
                                                                                         image_shape[2]),
                               name="conv2D_1"),
        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_1"),
        tf.keras.layers.Conv2D(64, 5, activation=tf.keras.activations.relu, name="conv2D_2"),
        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_2"),
        tf.keras.layers.Flatten(name="flatten_1"),
        tf.keras.layers.Dense(1, name="dense_1")
    ])

    model.summary()

    model.load_weights('weights.h5')

    plot_conv_weights(model, "conv2D_1")
    plot_conv_weights(model, "conv2D_2")
    plot_conv_weights(model, "dense_1")
