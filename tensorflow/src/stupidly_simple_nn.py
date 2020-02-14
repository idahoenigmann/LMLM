import tensorflow as tf
import numpy as np
import loadData
import os.path
import PIL
import matplotlib.pyplot as plt

SHOW_IMAGES = False
PREDICT = True
MODEL_IMAGE = False
NORMALIZE_OUTPUT = False
DRAW_LOSS = True

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

    sgd = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0, nesterov=False)

    model.summary()

    if MODEL_IMAGE:
        tf.keras.utils.plot_model(model, to_file='model.png')

    model.compile(loss=tf.keras.losses.mean_absolute_error, metrics=[tf.keras.losses.mean_absolute_percentage_error],
                  optimizer=sgd)

    if os.path.isfile('weights/weights.h5'):
        model.load_weights('weights/weights.h5')

    print("total image count: {}".format(loadData.get_image_count()))

    batch_size = 20
    cnt_batch = 2500 / batch_size
    percentage = int(loadData.get_image_count() / cnt_batch)
    i = 0

    hl, = plt.plot([], [])          # TODO

    def update_line(hl, new_data):
        hl.set_xdata(np.append(hl.get_xdata(), new_data))
        hl.set_ydata(np.append(hl.get_ydata(), new_data))
        plt.draw()

    while True:
        images = []
        labels = []

        print("-" * 100)
        print("images {} to {}:".format(int(i * percentage), int((i + 1) * percentage)))
        for file in loadData.load_images()[int(i * percentage):int((i + 1) * percentage)]:
            img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
            images.append(img)
            labels.append(label)

            if SHOW_IMAGES:
                imgplt = plt.imshow(img)
                plt.show()

        images_np = np.asarray(images, dtype=np.float)
        labels_np = np.asarray(labels)

        labels_x = tf.reshape(labels_np, [3 * batch_size])[0::3]
        labels_y = tf.reshape(labels_np, [3 * batch_size])[1::3]
        labels_z = tf.reshape(labels_np, [3 * batch_size])[2::3]

        model.fit(images_np, labels_np, epochs=1, verbose=1, steps_per_epoch=None)

        model.save_weights('weights/weights.h5')

        if PREDICT:
            prediction = model.predict(images_np)

            for idx in range(len(prediction)):
                if NORMALIZE_OUTPUT:
                    actual_un = labels_np[idx]
                    prediction_un = prediction[idx]
                else:
                    actual_un = loadData.unnormalize(labels_np[idx][0], labels_np[idx][1], labels_np[idx][2])
                    prediction_un = loadData.unnormalize(prediction[idx][0], prediction[idx][1], prediction[idx][2])

                print("idx - {:2}     prediction - [{:0<8.5f} {:0<8.5f} {:0<8.5f}]    ".format(idx, prediction_un[0],
                                                                                               prediction_un[1],
                                                                                               prediction_un[2]) \
                      + "actual - [{:0<8.5f} {:0<8.5f} {:0<8.5f}]".format(actual_un[0], actual_un[1], actual_un[2]))

        i = (i + 1) % cnt_batch
