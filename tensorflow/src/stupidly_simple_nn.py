import tensorflow as tf
import numpy as np
import loadData

if __name__ == '__main__':
    images = []
    labels = []
    for file in loadData.load_images():
        img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
        images.append(img)
        labels.append(label)

    images_np = np.asarray(images)
    labels_np = np.asarray(labels)

    labels_np = tf.reshape(labels_np, [images_np.shape[0], 1])

    print("image: {}; label: {}".format(images_np.shape, labels_np.shape))

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 5, activation="relu", input_shape=(images_np.shape[1], images_np.shape[2],
                                                                      images_np.shape[3]), name="conv2D_1"),
        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_1"),
        tf.keras.layers.Conv2D(64, 5, activation="relu", name="conv2D_2"),
        tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_2"),
        tf.keras.layers.Flatten(name="flatten_1"),
        tf.keras.layers.Dense(1, name="dense_1")
    ])
    model.summary()

    model.compile(loss="mean_squared_error",
        optimizer="SGD",
        metrics=['accuracy'])

    print("Model fit: {}".format(model.fit(images_np, labels_np, steps_per_epoch=10)))
    # print("Model test: {}".format(model.test_on_batch(images_np, labels_np)))
