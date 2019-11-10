import tensorflow as tf
import numpy as np
import pathlib
from keras.preprocessing.image import load_img, img_to_array

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
    return float(label) / 10.0


def decode_img(img, width, height):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    print(img.shape)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.reshape(img, [width * height, 1])
    return img


def process_path(file_path):
    label = get_label(file_path)

    img = load_img(file_path + ".jpg", grayscale=True)
    img = img_to_array(img)

    return img, label


def next_batch(num, images, labels):
    idx = np.arange(0, len(images))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [images[i].eval() for i in idx]
    labels_shuffle = [[labels[i]] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
