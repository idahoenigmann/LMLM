import tensorflow as tf
import pathlib

DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
IMG_WIDTH = 1920/5
IMG_HEIGHT = 1080/5


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin='file:///home/ida/Documents/5_Klasse/DA/LMLM/blender/data/renders/Suzanne.M_greyscale.tar.gz',
        fname='grayscale_renders', untar=True)
    data_dir = pathlib.Path(data_dir)

    """data_dir = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        fname='flower_photos', untar=True)
    data_dir = pathlib.Path(data_dir)"""

    image_count = len(list(data_dir.glob('*.jpg')))
    return list(data_dir.glob('*.jpg'))


def get_label(file_path):
    file = open("{}.txt".format(file_path), "w+")
    label = file.readline()
    file.close()
    return label


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return img


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path + ".jpg")
    img = decode_img(img)
    return img, label


if __name__ == '__main__':
    for file in load_images():
        process_path(str(file)[(str(file).find("."))])
