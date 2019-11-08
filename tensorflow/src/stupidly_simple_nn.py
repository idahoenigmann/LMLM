import tensorflow as tf
import loadData

if __name__ == '__main__':
    images = []
    labels = []
    for file in loadData.load_images():
        img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
        images.append(img)
        labels.append(label)

    """
    images_t = tf.constant(images, shape=(100, 20736, 1))
    labels_t = tf.constant(labels, shape=(100,))
    
    images_t[0, 0, 0] = 
    """

    # print("image: {}; label: {}".format(images[0].shape, len(labels)))

    model = tf.keras.Sequential()
    # dense_layer = tf.keras.layers.Dense(1, input_shape=(192 * 108, 1))
    dense_layer = tf.keras.layers.Dense(1, input_shape=(100, 192 * 108, 1))
    # dense_layer = tf.keras.layers.Dense(1, input_dim=1)
    model.add(dense_layer)

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
        optimizer="SGD",
        metrics=['accuracy'])

    print(model.fit(images, labels, steps_per_epoch=10, batch_size=10))
