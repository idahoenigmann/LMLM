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

    images_np = tf.reshape(images_np, [100, 108 * 192])
    labels_np = tf.reshape(labels_np, [100, 1])

    print("image: {}; label: {}".format(images_np.shape, labels_np.shape))

    model = tf.keras.Sequential()
    dense_layer = tf.keras.layers.Dense(1, input_shape=(images_np.shape[1],))

    model.add(dense_layer)
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
        optimizer="SGD",
        metrics=['accuracy'])

    print("Model fit: {}".format(model.fit(images_np, labels_np, steps_per_epoch=10)))

    print("Model test: {}".format(model.test_on_batch(images_np, labels_np)))
