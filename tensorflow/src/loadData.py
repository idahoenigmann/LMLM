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

origin = 'file:///home/lord/.keras/datasets/suzanne_all_1k.zip'
fname = 'suzanne_all_1k'

"""
origin = 'file:///home/lord/.keras/datasets/vase_all.zip'
fname = 'vase_all'"""


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    """res1 = list(data_dir.glob('?.jpg'))
    res1.sort()
    res2 = list(data_dir.glob('??.jpg'))
    res2.sort()
    res3 = list(data_dir.glob('???.jpg'))
    res3.sort()

    res = list()

    for e in res1:
        res.append(e)
    for e in res2:
        res.append(e)
    for e in res3:
        res.append(e)"""

    return list(data_dir.glob('*.jpg'))


def normalize(x, y, z):
    x = float(x) / 10.0
    y = float(y) / 16.0 + 0.5
    z = float(z) / 10.0 + 0.5
    return [x, y, z]


def unnormalize(x, y, z):
    x = float(x) * 10.0
    y = (float(y) - 0.5) * 16.0
    z = (float(z) - 0.5) * 10.0
    return [x, y, z]


def get_label(file_path):
    file = open("{}-2.txt".format(file_path), "r")
    x, y, z = normalize(float(file.readline()), float(file.readline()), float(file.readline()))
    file.close()
    return [x, y, z]


def process_path(file_path):
    label = get_label(file_path)

    img = load_img(file_path + ".jpg", color_mode="rgb")
    img = np.array(img)

    img = img / 255.0

    return img, label


def get_image_dimensions(file_path):
    img = load_img(file_path + ".jpg", color_mode="rgb")
    img = img_to_array(img)

    return img.shape


def get_image_count():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*.jpg')))
    return image_count
