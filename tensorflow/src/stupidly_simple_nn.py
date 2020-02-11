import tensorflow as tf
import numpy as np
import loadData
import os.path
import PIL
import matplotlib.pyplot as plt

SHOW_IMAGES = False
PREDICT = True
MODEL_IMAGE = False


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
        tf.keras.layers.Conv2D(32, 4, activation=tf.keras.activations.relu, name="conv2D_3"),
        # tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_3"),
        tf.keras.layers.Flatten(name="flatten_1"),
        # tf.keras.layers.Dense(128, name="dense_1"),
        tf.keras.layers.Dense(3, name="dense_2")
    ])

    sgd = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, nesterov=True)

    model.summary()

    if MODEL_IMAGE:
        tf.keras.utils.plot_model(model, to_file='model.png')

    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=sgd)

    if os.path.isfile('weights/weights.h5'):
        model.load_weights('weights/weights.h5')

    print("total image count: {}".format(loadData.get_image_count()))

    batch_size = 4
    cnt_batch = 2500 / batch_size
    percentage = int(loadData.get_image_count() / cnt_batch)
    i = 0

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

        model.fit(images_np, labels_np, epochs=3, verbose=2, steps_per_epoch=None)

        model.save_weights('weights/weights.h5')

        if PREDICT:
            prediction = model.predict(images_np)

            for idx in range(len(prediction)):
                print(
                    "idx - {}     prediction - {}    actual - {}".format(idx, prediction[idx], labels_np[idx]))

        i = (i + 1) % cnt_batch
