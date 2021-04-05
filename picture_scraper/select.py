import numpy as np
from PIL import Image
import os
from shutil import copy
from pathlib import Path
from numpy.random import permutation
import sys
os.chdir(sys.argv[1])

newfolder = 'exterior'

if not os.path.isdir(os.path.join(Path(os.getcwd()), newfolder)):
    os.mkdir(os.path.join(Path(os.getcwd()), newfolder))


def crop(filename, w, h):
    image = Image.open(filename)
    width = image.size[0]
    height = image.size[1]

    aspect = width / float(height)

    ideal_width = w
    ideal_height = h

    ideal_aspect = ideal_width / float(ideal_height)

    if aspect > ideal_aspect:
        new_width = int(ideal_aspect * height)
        offset = (width - new_width) / 2
        resize = (offset, 0, width - offset, height)
    else:
        new_height = int(width / ideal_aspect)
        offset = (height - new_height) / 2
        resize = (0, offset, width, height - offset)

    img = image.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)

    return np.array(img)


if __name__ == '__main__':
    print('%s started running.' % os.path.basename(__file__))
    for pic in permutation(os.listdir(os.path.join(os.getcwd(), 'pictures'))):
        im = crop(os.path.join(Path(os.getcwd()), 'pictures', pic), 100, 60)

        if np.all([
            np.greater(np.mean(im), 80),
            np.greater(np.std(im), 62),
            np.greater(np.median(im), 60),
            np.greater(np.quantile(im, .8), 125)
        ]):
            copy(os.path.join(Path(os.getcwd()), 'pictures', pic),
                 os.path.join(Path(os.getcwd()), newfolder))
    print('%s finished running.' % os.path.basename(__file__))
