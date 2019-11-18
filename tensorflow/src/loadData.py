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


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin='file:///home/ida/.keras/datasets/suzanne_m_10k.tar.gz',
        fname='suzanne_m_50', untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*.jpg')))
    print("found {} images in path {}".format(image_count, data_dir))

    return list(data_dir.glob('*.jpg'))


def get_label(file_path):
    file = open("{}.txt".format(file_path), "r")
    label = file.read()
    file.close()
    return float(label) / 10.0


def process_path(file_path):
    label = get_label(file_path)

    img = load_img(file_path + ".jpg", grayscale=True)
    img = img_to_array(img)

    return img, label
