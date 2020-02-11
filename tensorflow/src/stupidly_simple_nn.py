import tensorflow as tf
import numpy as np
import loadData
import os.path
import PIL
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = loadData.load_images()[0]
    image_shape = loadData.get_image_dimensions(str(file)[0:(str(file).rfind("."))])

    input = tf.keras.Input(shape=(image_shape[0], image_shape[1], image_shape[2]))
    hidden_conv_0 = tf.keras.layers.Conv2D(32, 5, activation=tf.keras.activations.relu, name="conv2D_1")(input)
    hidden_pool_1 = tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_1")(hidden_conv_0)
    hidden_conv_2 = tf.keras.layers.Conv2D(64, 5, activation=tf.keras.activations.relu, name="conv2D_2")(hidden_pool_1)
    hidden_pool_3 = tf.keras.layers.MaxPooling2D(10, name="maxPooling2D_2")(hidden_conv_2)
    hidden_conv_4 = tf.keras.layers.Conv2D(32, 4, activation=tf.keras.activations.relu, name="conv2D_3")(hidden_pool_3)
    hidden_flat_5 = tf.keras.layers.Flatten(name="flatten_1")(hidden_conv_4)
    output_x = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(hidden_flat_5)
    output_y = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(hidden_flat_5)
    output_z = tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)(hidden_flat_5)

    sgd = tf.keras.optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    model = tf.keras.Model(inputs=input, outputs=[output_x, output_y, output_z])

    model.summary()

    tf.keras.utils.plot_model(model, to_file='model.png')

    model.compile(loss=tf.keras.losses.mean_absolute_error, loss_weights=[1.0, 0.5, 0.5],
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9))

    print("total image count: {}".format(loadData.get_image_count()))

    batch_size = 4
    cnt_batch = 2500 / batch_size
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

        labels_np = tf.reshape(labels_np, [3 * batch_size])

        labels_x = labels_np[0::3]
        labels_y = labels_np[1::3]
        labels_z = labels_np[2::3]

        if os.path.isfile('weights/weights.h5'):
            model.load_weights('weights/weights.h5')

        # imgplt = plt.imshow(images_np[0])
        # plt.show()

        model.fit(images_np, [labels_x, labels_y, labels_z], epochs=3, verbose=2, steps_per_epoch=None)

        model.save_weights('weights/weights.h5')

        print(model.predict(np.array(images_np)))

        """idx = 0
        for estimated in model.predict(images_np):
            print("actual: {} estimated: {}".format(np.array([labels_x[idx], labels_y[idx], labels_z[idx]]), estimated))
            idx += 1"""

        i = (i + 1) % cnt_batch
