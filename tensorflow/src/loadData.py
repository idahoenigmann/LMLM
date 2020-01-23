import tensorflow as tf
import numpy as np
import pathlib
from keras.preprocessing.image import load_img, img_to_array

"""
  _                     _   _____        _        
 | |                   | | |  __ \      | |       
 | |     ___   __ _  __| | | |  | | __ _| |_ __ _ 
 | |    / _ \ / _` |/ _` | | |  | |/ _` | __/ _` |
 | |___| (_) | (_| | (_| | | |__| | (_| | || (_| |
 |______\___/ \__,_|\__,_| |_____/ \__,_|\__\__,_|

"""

origin = 'file:///home/ida/.keras/datasets/suzanne_l_1k.tar.gz'
fname = 'suzanne_l_1k'


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)
    return list(data_dir.glob('*.jpg'))


def get_label(file_path):
    file = open("{}-2.txt".format(file_path), "r")
    x = float(file.readline()) / 10.0
    y = float(file.readline()) / 10.0
    z = float(file.readline()) / 10.0
    file.close()
    return [x, y, z]


def process_path(file_path):
    label = get_label(file_path)

    img = load_img(file_path + ".jpg", color_mode="grayscale")
    img = img_to_array(img)

    return img, label


def get_image_dimensions(file_path):
    img = load_img(file_path + ".jpg", color_mode="grayscale")
    img = img_to_array(img)

    return img.shape


def get_image_count():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*.jpg')))
    return image_count
