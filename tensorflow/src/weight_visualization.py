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

    """if len(W.shape) == 2:
        W = W.reshape((4 * 18, 64))
        plt.imshow(W)

        plt.axis('off')
        plt.tick_params(axis='both', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
"""
    if len(W.shape) == 2:
        W = W.reshape((4, 18, 64))

        fig, axs = plt.subplots(4, 8, figsize=(18, 2))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        axs = axs.ravel()
        for i in range(32):
            axs[i].imshow(W[:, :, i])
            axs[i].set_xticks([])
            axs[i].set_yticks([])

    if len(W.shape) == 4:
        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3]))
        fig, axs = plt.subplots(4, 8, figsize=(16, 8))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        axs = axs.ravel()
        for i in range(32):
            axs[i].imshow(W[:, :, i])
            axs[i].set_xticks([])
            axs[i].set_yticks([])

    plt.savefig('implementation_neuralNetwork_structureOfOurNeuralNetwork_' + layer + '.png')
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

    # model.load_weights('pretrained_weights_suzanne_greyscale_m.h5')
    model.load_weights('pretrained_weights_suzanne_greyscale_m_momentum.h5')
    # model.load_weights('pretrained_weights_suzanne_greyscale_m_momentum2.h5')
    # model.load_weights('weights.h5')

    plot_conv_weights(model, "conv2D_1")
    plot_conv_weights(model, "conv2D_2")
    plot_conv_weights(model, "dense_1")

    """
    # yellow = high number
    # blue = low number
    test = np.zeros((10, 2))

    for i in range(0, 10):
        test[i, 0] = i / 10

    test[9, 1] = 0.9

    print(test)

    plt.imshow(test)
    plt.show()
    """
