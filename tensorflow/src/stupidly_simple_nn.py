import tensorflow as tf
import numpy as np
import loadData
import os.path

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
                  metrics=[tf.keras.metrics.mean_absolute_error])

    print("total image count: {}".format(loadData.get_image_count()))

    cnt_batch = 15
    percentage = int(loadData.get_image_count() / cnt_batch)
    i = 0   # // percentage        # start image count
    while True:
        images = []
        labels = []

        print("-" * 100)
        print("from {} to {}".format(i * percentage, (i + 1) * percentage))
        for file in loadData.load_images()[i * percentage:(i + 1) * percentage]:
            img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
            images.append(img)
            labels.append(label)

        images_np = np.asarray(images)
        labels_np = np.asarray(labels)

        labels_np = tf.reshape(labels_np, [images_np.shape[0], 3])

        if os.path.isfile('weights/weights.h5'):
            model.load_weights('weights/weights.h5')

        model.fit(images_np, labels_np, epochs=3, verbose=2)

        model.save_weights('weights/weights.h5')

        for idx in range(len(images_np)):
            shape = list(images_np[idx].shape)
            shape.insert(0, 1)

            img = images_np[-1].reshape(shape)

            print("expected: {}; actual: {}".format(labels_np[idx], model.predict(img)))

        i = (i + 1) % cnt_batch
