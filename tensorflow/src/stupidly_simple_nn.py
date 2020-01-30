import tensorflow as tf
import numpy as np
import loadData
import os.path
import PIL
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = loadData.load_images()[0]
    image_shape = loadData.get_image_dimensions(str(file)[0:(str(file).rfind("."))])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 5, activation=tf.keras.activations.relu, input_shape=(image_shape[0], image_shape[1],
                                                                      image_shape[2]), name="conv2D_1"),
        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_1"),
        tf.keras.layers.Conv2D(64, 5, activation=tf.keras.activations.relu, name="conv2D_2"),
        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_2"),
        tf.keras.layers.Conv2D(32, 4, activation=tf.keras.activations.relu, name="conv2D_3"),
        # tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_3"),
        tf.keras.layers.Flatten(name="flatten_1"),
        # tf.keras.layers.Dense(128, name="dense_1"),
        tf.keras.layers.Dense(3, name="dense_2")
    ])
    model.summary()

    # tf.keras.utils.plot_model(model, to_file='model.png')

    model.compile(loss=tf.keras.losses.mean_absolute_error,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=[tf.keras.metrics.mean_absolute_percentage_error])

    print("total image count: {}".format(loadData.get_image_count()))

    cnt_batch = 2500 / 4
    percentage = int(loadData.get_image_count() / cnt_batch)
    i = 0

    while True:
        images = []
        labels = []

        print("-" * 100)
        print("from {} to {}".format(int(i * percentage), int((i + 1) * percentage)))
        for file in loadData.load_images()[int(i * percentage):int((i + 1) * percentage)]:
            img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
            images.append(img)
            labels.append(label)

        images_np = np.asarray(images, dtype=np.float)
        labels_np = np.asarray(labels)

        labels_np = tf.reshape(labels_np, [images_np.shape[0], 3])

        if os.path.isfile('weights/weights.h5'):
            model.load_weights('weights/weights.h5')

        # imgplt = plt.imshow(images_np[0])
        # plt.show()

        model.fit(images_np, labels_np, epochs=3, verbose=2, steps_per_epoch=10)

        model.save_weights('weights/weights.h5')

        idx = 0
        for actual in model.predict_on_batch(images_np):
            print("expected: {}; actual: {}".format(labels_np[idx], actual))
            idx += 1

        i = (i + 1) % cnt_batch
