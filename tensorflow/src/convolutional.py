import tensorflow as tf
import numpy as np
import pathlib

IMG_WIDTH = 192
IMG_HEIGHT = 108
STEPS = 10

"""
   _____      _                 _   _ _   _ 
  / ____|    | |               | \ | | \ | |
 | (___   ___| |_ _   _ _ __   |  \| |  \| |
  \___ \ / _ \ __| | | | '_ \  | . ` | . ` |
  ____) |  __/ |_| |_| | |_) | | |\  | |\  |
 |_____/ \___|\__|\__,_| .__/  |_| \_|_| \_|
                       | |                  
                       |_|                  
"""

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

"""
  _                     _   _____        _        
 | |                   | | |  __ \      | |       
 | |     ___   __ _  __| | | |  | | __ _| |_ __ _ 
 | |    / _ \ / _` |/ _` | | |  | |/ _` | __/ _` |
 | |___| (_) | (_| | (_| | | |__| | (_| | || (_| |
 |______\___/ \__,_|\__,_| |_____/ \__,_|\__\__,_|
                                                  
"""

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


def next_batch(num, images, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(images))
    np.random.shuffle(idx)
    idx = idx[:num]
    # print(images[0].eval().shape)
    # print(images[0].eval().reshape(1080 * 1920).shape)
    data_shuffle = [images[i].eval().reshape(IMG_HEIGHT * IMG_WIDTH) for i in idx]
    labels_shuffle = [[labels[i]] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


if __name__ == '__main__':
    """
   _____      _               
  / ____|    | |              
 | (___   ___| |_ _   _ _ __  
  \___ \ / _ \ __| | | | '_ \ 
  ____) |  __/ |_| |_| | |_) |
 |_____/ \___|\__|\__,_| .__/ 
                       | |    
                       |_|    
    """
    x = tf.placeholder(tf.float32, shape=[None, IMG_WIDTH * IMG_HEIGHT])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    x_image = tf.reshape(x, [-1, IMG_WIDTH, IMG_HEIGHT, 1])
    conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    # print(conv2_pool.shape)
    # conv2_flat = tf.reshape(conv2_pool, [-1, int((IMG_WIDTH * IMG_HEIGHT)/4) * 64])
    conv2_flat = tf.reshape(conv2_pool, [-1, int(IMG_WIDTH/4 * IMG_HEIGHT/4) * 64])
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

    keep_prob = tf.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full_1, rate=1 - keep_prob)

    y_conv = full_layer(full1_drop, 1)

    """
             _     _____        _        
            | |   |  __ \      | |       
   __ _  ___| |_  | |  | | __ _| |_ __ _ 
  / _` |/ _ \ __| | |  | |/ _` | __/ _` |
 | (_| |  __/ |_  | |__| | (_| | || (_| |
  \__, |\___|\__| |_____/ \__,_|\__\__,_|
   __/ |                                 
  |___/                                  
    """

    images = []
    labels = []
    for file in load_images():
        img, label = process_path(str(file)[0:(str(file).rfind("."))])
        images.append(img)
        labels.append(label)

    """
  _             _       
 | |           (_)      
 | |_ _ __ __ _ _ _ __  
 | __| '__/ _` | | '_ \ 
 | |_| | | (_| | | | | |
  \__|_|  \__,_|_|_| |_|
                                     
    """
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            batch = next_batch(5, images, labels)

            if i % 1 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step {}, training accuracy {}".format(i, train_accuracy))

            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        X, Y = next_batch(10, images, labels)
        test_accuracy = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob:1.0}) for i in range(10)])

    print("test accuracy: {}".format(test_accuracy))
