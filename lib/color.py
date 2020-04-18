from PIL import Image
import numpy as n
from scipy import ndimage, interpolate
from scipy.io import loadmat
import scipy.cluster
from lib.utils import *
import random

class ColourState(object):
    """
    Gives the foreground, background, and optionally border colourstate.
    Does this by sampling from a training set of images, and clustering in to desired number of colours
    (http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
    """
    IMFN = "/home/ubuntu/Datasets/text-renderer/image_24_results.png"

    def __init__(self):
        self.im = rgb2gray(n.array(Image.open(self.IMFN)))

    def get_sample(self, n_colours):
        #print('Inside Color State')
        a = self.im.flatten()
        codes, dist = scipy.cluster.vq.kmeans(a, n_colours)
        # get std of centres
        vecs, dist = scipy.cluster.vq.vq(a, codes)

        colours = []
        for i in range(n_colours):
            try:
                code = codes[i]
                std = n.std(a[vecs==i])
                colours.append(std*n.random.randn() + code)
            except IndexError:
                print("\tcolour error")
                colours.append(int(sum(colours)/float(len(colours))))
        # choose randomly one of each colour
        return n.random.permutation(colours)

class TrainingCharsColourState(object):
    """
    Gives the foreground, background, and optionally border colourstate.
    Does this by sampling from a training set of images, and clustering in to desired number of colours
    (http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
    """

    def __init__(self, list_fn):
        self.ims = []
        with open(list_fn) as f:
            self.ims += (f.read().splitlines())

    def get_sample(self, n_colours):
        curs = 0
        while True:
            curs += 1
            if curs > 1000:
                print("problem with colours")
                break

            imfn = random.choice(self.ims)
            im = rgb2gray(n.array(Image.open(imfn)))
            #im = self.ims[...,n.random.randint(0, self.ims.shape[2])]

            a = im.flatten()
            codes, dist = scipy.cluster.vq.kmeans(a, n_colours)
            if len(codes) != n_colours:
                continue
            # get std of centres
            vecs, dist = scipy.cluster.vq.vq(a, codes)
            colours = []
            for i, code in enumerate(codes):
                std = n.std(a[vecs==i])
                colours.append(std*n.random.randn() + code)
            break
        # choose randomly one of each colour
        return n.random.permutation(colours)