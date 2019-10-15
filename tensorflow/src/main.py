import tensorflow as tf
import numpy as np
import pathlib

NUM_STEPS = 10
MINIBATCH_SIZE = 5
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin='file:///home/ida/Documents/5_Klasse/DA/LMLM/blender/data/renders/Suzanne.M_greyscale.tar.gz',
        fname='grayscale_renders', untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*.jpg')))
    print("found {} images in path {}".format(image_count, data_dir))

    return list(data_dir.glob('*.jpg'))


def get_label(file_path):
    file = open("{}.txt".format(file_path), "r")
    label = file.read()
    file.close()
    return label


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path + ".jpg")
    img = decode_img(img)
    return img, label


def setup(width=IMG_WIDTH, height=IMG_HEIGHT):
    x = tf.placeholder(tf.float32, [None, width * height])  # size of pictures?
    W = tf.Variable(tf.zeros([width * height, 1]))

    y_true = tf.placeholder(tf.float32, [None, 1])
    y_pred = tf.matmul(x, W)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=y_pred, labels=y_true))

    gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    return gd_step, accuracy, y_true, x


def next_batch(num, images, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(images))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [images[i].eval() for i in idx]
    labels_shuffle = [[labels[i]] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train_and_test(gd_step, accuracy, y_true, x, images, labels):
    with tf.Session() as sess:
        # Train
        sess.run(tf.global_variables_initializer())

        for i in range(NUM_STEPS):
            batch_xs, batch_ys = next_batch(MINIBATCH_SIZE, images, labels)
            # print("batch_xs: {}\nbatch_ys: {}".format(batch_xs, batch_ys))
            sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

        # Test
        return sess.run(accuracy, feed_dict={x: images, y_true: labels})


if __name__ == '__main__':
    images = []
    labels = []
    for file in load_images():
        img, label = process_path(str(file)[0:(str(file).rfind("."))])
        images.append(img)
        labels.append(label)

    gd_step, accuracy, y_true, x = setup()
    print("Accuracy: {:.4}%".format(train_and_test(gd_step, accuracy, y_true, x, images, labels) * 100))
