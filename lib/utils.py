import numpy as n
import pygame
from PIL import Image
import sys

def wait_key():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.localsKEYDOWN and event.key == pygame.locals.K_SPACE:
                return


def save_screen_img(pg_surface, fn, quality=100):
    imgstr = pygame.image.tostring(pg_surface, 'RGB')
    im = Image.frombytes('RGB', pg_surface.get_size(), imgstr)
    im.save(fn, quality=quality)
    print("save path: ", fn)


def rgb2gray(rgb):
    # RGB -> grey-scale (as in Matlab's rgb2grey)
    try:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    except IndexError:
        try:
            gray = rgb[:,:,0]
        except IndexError:
            gray = rgb[:,:]
    return gray


def resize_image(im, r=None, newh=None, neww=None, filtering=Image.BILINEAR):
    dt = im.dtype
    I = Image.fromarray(im)
    if r is not None:
        h = im.shape[0]
        w = im.shape[1]
        newh = int(round(r*h))
        neww = int(round(r*w))
    if neww is None:
        neww = int(newh*im.shape[1]/float(im.shape[0]))
    if newh > im.shape[0]:
        I = I.resize([neww, newh], Image.ANTIALIAS)
    else:
        I.thumbnail([neww, newh], filtering)
    return n.array(I).astype(dt)


def get_bb(arr, eq=None):
    if eq is None:
        v = n.nonzero(arr > 0)
    else:
        v = n.nonzero(arr == eq)
    xmin = v[1].min()
    xmax = v[1].max()
    ymin = v[0].min()
    ymax = v[0].max()
    return [xmin, ymin, xmax-xmin, ymax-ymin]

def get_rects_union_bb(rects, arr):
    rectarr = n.zeros((arr.shape[0], arr.shape[1]))
    for i, rect in enumerate(rects):
        starti = max(0, rect[1])
        endi = min(rect[1]+rect[3], rectarr.shape[0])
        startj = max(0, rect[0])
        endj = min(rect[0]+rect[2], rectarr.shape[1])
        rectarr[starti:endi, startj:endj] = 10
    return get_bb(rectarr)

def resize_rects(rects, arr, outheight):
    rectarr = n.zeros((arr.shape[0], arr.shape[1]))
    for i, rect in enumerate(rects):
        starti = max(0, rect[1])
        endi = min(rect[1]+rect[3], rectarr.shape[0])
        startj = max(0, rect[0])
        endj = min(rect[0]+rect[2], rectarr.shape[1])
        rectarr[starti:endi, startj:endj] = (i+1)*10
    rectarr = resize_image(rectarr, newh=outheight, filtering=Image.NONE)
    newrects = []
    for i, _ in enumerate(rects):
        try:
            newrects.append(pygame.Rect(get_bb(rectarr, eq=(i+1)*10)))
        except ValueError:
            pass
    return newrects


def imcrop(arr, rect):
    if arr.ndim > 2:
        return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2],...]
    else:
        return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]


def invert_arr(arr):
    arr ^= 2 ** 32 - 1
    return arr


def invert_surface(surf):
    pixels = pygame.surfarray.pixels2d(surf)
    pixels ^= 2 ** 32 - 1
    del pixels


def arr_scroll(arr, dx, dy):
    arr = n.roll(arr, dx, axis=1)
    arr = n.roll(arr, dy, axis=0)
    return arr

def stack_arr(arrs):
    shp = list(arrs[0].shape)
    shp.append(1)
    tup = []
    for arr in arrs:
        tup.append(arr.reshape(shp))
    return n.concatenate(tup, axis=2)