from PIL import Image
import random
import numpy as n
from lib.utils import *

class FillImageState(object):
    """
    Handles the images used for filling the background, foreground, and border surfaces
    """
    MJBLEND_NORMAL = "normal"
    MJBLEND_ADD = "add"
    MJBLEND_SUB = "subtract"
    MJBLEND_MULT = "multiply"
    MJBLEND_MULTINV = "multiplyinv"
    MJBLEND_SCREEN = "screen"
    MJBLEND_DIVIDE = "divide"
    MJBLEND_MIN = "min"
    MJBLEND_MAX = "max"

    DATA_DIR = '/home/ubuntu/Pictures/'
    IMLIST = ['maxresdefault.jpg', 'alexis-sanchez-arsenal-wallpaper-phone.jpg', 'alexis.jpeg']
    blend_amount = [0.0, 0.25]  # normal dist mean, std
    blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
    blend_order = 0.5
    min_textheight = 16.0  # minimum pixel height that you would find text in an image

    def get_sample(self, surfarr):
        """
        The image sample returned should not have it's aspect ratio changed, as this would never happen in real world.
        It can still be resized of course.
        """
        # load image
        imfn = random.choice(self.IMLIST)
        baseim = n.array(Image.open(imfn))
        # choose a colour channel or rgb2gray
        if baseim.ndim == 3:
            if n.random.rand() < 0.25:
                baseim = rgb2gray(baseim)
            else:
                baseim = baseim[...,n.random.randint(0,3)]
        else:
            assert(baseim.ndim == 2)

        imsz = baseim.shape
        surfsz = surfarr.shape

        # don't resize bigger than if at the original size, the text was less than min_textheight
        max_factor = float(surfsz[0])/self.min_textheight
        # don't resize smaller than it is smaller than a dimension of the surface
        min_factor = max(float(surfsz[0] + 5)/float(imsz[0]), float(surfsz[1] + 5)/float(imsz[1]))
        # sample a resize factor
        factor = max(min_factor, min(max_factor, ((max_factor-min_factor)/1.5)*n.random.randn() + max_factor))
        sampleim = resize_image(baseim, factor)
        imsz = sampleim.shape
        # sample an image patch
        good = False
        curs = 0
        while not good:
            curs += 1
            if curs > 1000:
                print("difficulty getting sample")
                break
            try:
                x = n.random.randint(0,imsz[1]-surfsz[1])
                y = n.random.randint(0,imsz[0]-surfsz[0])
                good = True
            except ValueError:
                # resample factor
                factor = max(min_factor, min(max_factor, ((max_factor-min_factor)/1.5)*n.random.randn() + max_factor))
                sampleim = resize_image(baseim, factor)
                imsz = sampleim.shape

        imsample = (n.zeros(surfsz) + 255).astype(surfarr.dtype)
        imsample[...,0] = sampleim[y:y+surfsz[0],x:x+surfsz[1]]
        imsample[...,1] = surfarr[...,1].copy()

        return {
            'image': imsample,
            'blend_mode': random.choice(self.blend_modes),
            'blend_amount': min(1.0, n.abs(self.blend_amount[1]*n.random.randn() + self.blend_amount[0])),
            'blend_order': n.random.rand() < self.blend_order,
        }


class SVTFillImageState(FillImageState):
    IMLIST=[]
    def __init__(self, list_fn):
        with open(list_fn) as f:
            self.IMLIST += (f.read().splitlines())