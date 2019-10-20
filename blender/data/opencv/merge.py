import cv2
from argparse import ArgumentParser
import os
import numpy as np

parser = ArgumentParser(description='Merges images from root directory')
parser.add_argument('-d', help='the directory to traverse',
                    metavar='DIRECTORY', dest='d', type=str)
args = parser.parse_args()


for dirname, subdirlist, filelist in os.walk(args.d):
    if dirname.find("_merged") != -1 or dirname == args.d:
        print("Skipping {} !".format(dirname))
        continue
    
    newdir = "{}_merged".format(dirname)
    try:
        os.mkdir(newdir)
    except OSError:
        pass
    print("Writing to: {}".format(newdir))
    counter = 0
    for i in range(0, len(filelist), 2):
        path1 = "{}/{}".format(dirname, filelist[i])
        path2 = "{}/{}".format(dirname, filelist[i + 1])

        print("{} | {}".format(path1, path2))

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        merged = np.concatenate((img1, img2), axis=1)

        cv2.imwrite("{}/{}.jpg".format(newdir, counter), merged, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        counter += 1




