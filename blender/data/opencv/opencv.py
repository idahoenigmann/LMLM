import cv2
import os
import numpy as np
from argparse import ArgumentParser

# -d ../renders/Suzanne.M -o saturated
options = ['greyscale', 'saturated', 'downscaled', 'cropping']

parser = ArgumentParser(description='Manipulate images in directory')
parser.add_argument('-d', help='the directory to process',
                    metavar='DIRECTORY', dest='d', type=str)
parser.add_argument('-o', help='how to manipulate the images',
                    metavar='OPTION', dest='o', type=str)

args = parser.parse_args()
if args.o not in options:
    print("Unknown option")
    exit(1)

newdir_path = list(filter(None, args.d.split('/')))
newdir_path[-1] += ('_{}'.format(args.o))
newdir_path = '/'.join(newdir_path)

try:
    os.mkdir(newdir_path)
except OSError:
    print('Found {}'.format(newdir_path))
    
print('Will create files in: {}'.format(newdir_path))
contents = os.listdir(args.d)
for file in contents:
    filename = os.fsdecode(file)
    if filename.endswith('.jpg'):
        image = cv2.imread('{}/{}'.format(args.d, filename))

        if args.o == options[0]:    # greyscale
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('{}/{}'.format(newdir_path, filename), grey, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        elif args.o == options[1]:    # saturated
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float32')
            (h, s, v) = cv2.split(hsv)
            s *= 1.5
            s = np.clip(s, 0, 255)
            hsv = cv2.merge([h, s, v])
            saturated = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
            cv2.imwrite('{}/{}'.format(newdir_path, filename), saturated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        elif args.o == options[2]:    # downscaled
            scale_percent = 10  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            downscaled = cv2.resize(image, dim)
            cv2.imwrite('{}/{}'.format(newdir_path, filename), downscaled, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        elif args.o == options[3]:    # cropping
            crop_margin_percent = 5
            crop_margin_width = int(image.shape[1] * crop_margin_percent / 100)
            crop_margin_height = int(image.shape[0] * crop_margin_percent / 100)
            crop_img = image[crop_margin_width:-crop_margin_width, crop_margin_height:-crop_margin_height]
            cv2.imwrite('{}/{}'.format(newdir_path, filename), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        else:
            print('Unknown option: {}'.format(args.o))
            print('Option has to be one of the following: ')
            print(options)
            exit(1)
