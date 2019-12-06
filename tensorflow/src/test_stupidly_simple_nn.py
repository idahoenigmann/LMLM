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
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=[tf.keras.metrics.mean_absolute_percentage_error])

    print("total image count: {}".format(loadData.get_image_count()))

    images = []
    labels = []

    for file in loadData.load_images()[0:20]:
        img, label = loadData.process_path(str(file)[0:(str(file).rfind("."))])
        images.append(img)
        labels.append(label)

    images_np = np.asarray(images)
    labels_np = np.asarray(labels)

    model.load_weights('weights/weights.h5')

    labels_estimated_np = model.predict(images_np).flatten()

    absolute_average = 0
    absolute_percentage_average = 0

    print("{:15}  {:15}  {:15}  {:15}".format("real", "estimated", "diff", "percentage"))

    for i in range(len(labels_np)):
        absolute_average += abs(labels_np[i] - labels_estimated_np[i])
        absolute_percentage_average += abs(100 - (labels_estimated_np[i]/labels_np[i]*100))

        print("{:15.7}m {:15.7}m {:15.7}m {:15.7}%".format(labels_np[i], labels_estimated_np[i],
                                                     labels_np[i] - labels_estimated_np[i],
                                                     labels_estimated_np[i]/labels_np[i]*100))

    absolute_average = absolute_average / len(labels_np)
    absolute_percentage_average = absolute_percentage_average / len(labels_np)

    print("\nabsolute average: {:.7}m".format(absolute_average))
    print("absolute percentage average: {:.7}%".format(absolute_percentage_average))
