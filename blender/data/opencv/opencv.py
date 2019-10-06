import cv2
import os
from argparse import ArgumentParser

# -d renders/Suzanne.M -o greyscale
options = ['greyscale']

parser = ArgumentParser(description='Manipulate images in directory')
parser.add_argument('-d', help='the directory to process',
                    metavar='DIRECTORY', dest='d', type=str)
parser.add_argument('-o', help='how to manipulate the images',
                    metavar='OPTION', dest='o', type=str)

try:
    args = parser.parse_args()
    if args.o not in options:
        raise

    newdir_path = list(filter(None, args.d.split('/')))
    newdir_path[-1] += ('_{}'.format(args.o))
    newdir_path = '/'.join(newdir_path)
    os.mkdir(newdir_path)
    print('Will create files in: {}'.format(newdir_path))
    for file in os.listdir(args.d):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg'):
            image = cv2.imread('{}/{}'.format(args.d, filename))

            if args.o == options[0]:
                grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('{}/{}'.format(newdir_path, filename), grey, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
         
except:
    print('First argument has to be a valid path\nSecond argument has to be one of the following:')
    print(options)
    exit(1)

