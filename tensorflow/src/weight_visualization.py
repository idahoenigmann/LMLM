import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
import primefactors
import math


def plot_conv_weights(model, layer, shape, landscape=False):
    W = model.get_layer(name=layer).get_weights()[0]
    print("shape of layer {}: {} will be reshaped to {}".format(layer, W.shape, shape))
    W = W.reshape(shape)

    if len(shape) == 3:
        a, b = 1, 1
        even = True
        for e in primefactors.factorize(shape[-1]):
            if even:
                a *= e
            else:
                b *= e
            even = not even

        if landscape:
            a, b = b, a

        gcd = math.gcd(b * shape[1], a * shape[0])

        konst_mul = 1

        if ((a * shape[0] / gcd) <= 2) or ((b * shape[1] / gcd) <= 2):     # avoid images smaller than 2
            konst_mul = 20

        print("figsize({}, {})".format(b * shape[1] / gcd * konst_mul, a * shape[0] / gcd * konst_mul))

        fig, axs = plt.subplots(a, b, figsize=(b * shape[1] / gcd * konst_mul,
                                               a * shape[0] / gcd * konst_mul))
        fig.subplots_adjust(hspace=0.2, wspace=0.1)

        axs = axs.ravel()

        for i in range(shape[-1]):
            axs[i].imshow(W[:, :, i]) #, vmin=0, vmax=1)
            axs[i].set_xticks([])
            axs[i].set_yticks([])

    elif len(shape) == 2:
        plt.imshow(W)
        plt.axis('off')
        plt.tick_params(axis='both', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.tight_layout(0, 0.1, 0.1)

    plt.savefig('implementation_neuralNetwork_structureOfOurNeuralNetwork_' + layer + '.svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    file = loadData.load_images()[0]
    image_shape = loadData.get_image_dimensions(str(file)[0:(str(file).rfind("."))])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 5, activation=tf.keras.activations.linear, input_shape=(image_shape[0],
                                                                                         image_shape[1],
                                                                                         image_shape[2]),
                               name="conv2D_1"),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="activation_1"),
        tf.keras.layers.MaxPooling2D(5, name="maxPooling2D_1"),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="activation_2"),

        tf.keras.layers.Conv2D(64, 5, activation=tf.keras.activations.linear, name="conv2D_2"),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="activation_3"),

        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_2"),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="activation_4"),

        tf.keras.layers.Conv2D(64, 5, activation=tf.keras.activations.linear, name="conv2D_3"),
        tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="activation_5"),

        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_3"),

        tf.keras.layers.Flatten(name="flatten_1"),

        tf.keras.layers.Dense(3, activation=tf.keras.activations.sigmoid, name="dense_1")
    ])

    model.summary()

    # model.load_weights('weights/pretrained_weights_suzanne_greyscale_m.h5')
    # model.load_weights('weights/pretrained_weights_suzanne_greyscale_m_momentum.h5')
    # model.load_weights('weights/pretrained_weights_suzanne_greyscale_m_momentum2.h5')
    model.load_weights('weights/weights.h5')

    # plot_conv_weights(model, "conv2D_1", (5, 5, 3 * 32))
    # plot_conv_weights(model, "conv2D_2", (5, 5, 32 * 64), True)
    # plot_conv_weights(model, "conv2D_3", (5, 5, 64 * 64))
    plot_conv_weights(model, "dense_1", (3, 3, 64))

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
