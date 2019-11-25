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
        tf.keras.layers.Flatten(name="flatten_1"),
        tf.keras.layers.Dense(1, name="dense_1")
    ])
    model.summary()

    model.compile(loss=tf.keras.losses.mean_absolute_error,
                  optimizer="SGD",
                  metrics=[tf.keras.metrics.mean_absolute_percentage_error])

    print("total image count: {}".format(loadData.get_image_count()))

    percentage = int(loadData.get_image_count() / 200)
    i = int(1000 / percentage)        # start image count
    while True:
        i = (i + 1) % percentage
    # for i in range(0, percentage):
        images = []
        labels = []

        print("from {} to {}".format(i * percentage, (i + 1) * percentage))
        for file in loadData.load_images()[i * percentage:(i + 1) * percentage]:
            img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
            images.append(img)
            labels.append(label)

        images_np = np.asarray(images)
        labels_np = np.asarray(labels)

        labels_np = tf.reshape(labels_np, [images_np.shape[0], 1])

        if os.path.isfile('weights.h5'):
            model.load_weights('weights.h5')

        print("Model fit: {}".format(model.fit(images_np, labels_np, batch_size=10, epochs=1,
                                               verbose=1, steps_per_epoch=10)))

        model.save_weights('weights.h5')
    # print("Model test: {}".format(model.test_on_batch(images_np[0:15], labels_np[0:15])))
