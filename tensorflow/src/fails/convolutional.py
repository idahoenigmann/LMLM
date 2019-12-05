import tensorflow as tf

from loadData import *
from setupNN import *

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

        #X, Y = next_batch(10, images, labels)
        #test_accuracy = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob:1.0}) for i in range(10)])

    print("test accuracy: {}".format(test_accuracy))