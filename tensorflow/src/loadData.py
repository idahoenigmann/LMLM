import tensorflow as tf
import numpy as np
import pathlib

IMG_WIDTH = 192
IMG_HEIGHT = 108
STEPS = 10

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
