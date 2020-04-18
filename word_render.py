import os, math, pygame, random, cv2
import numpy as n

from pygame.locals import *
from pygame import freetype
from scipy import ndimage, interpolate
from PIL import Image

from lib.transform import (
        PerspectiveTransformState,
        AffineTransformState,
        DistortionState,
        SurfaceDistortionState,
        ElasticDistortionState,
)
from lib.border_render import BorderState
from lib.font_render import BaselineState
from lib.utils import *


MJBLEND_NORMAL = "normal"
MJBLEND_ADD = "add"
MJBLEND_SUB = "subtract"
MJBLEND_MULT = "multiply"
MJBLEND_MULTINV = "multiplyinv"
MJBLEND_SCREEN = "screen"
MJBLEND_DIVIDE = "divide"
MJBLEND_MIN = "min"
MJBLEND_MAX = "max"


def grey_blit(src, dst, blend_mode=MJBLEND_NORMAL):
    """
    This is for grey + alpha images
    """
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    # blending with alpha http://stackoverflow.com/questions/1613600/direct3d-rendering-2d-images-with-multiply-blending-mode-and-alpha
    # blending modes from: http://www.linuxtopia.org/online_books/graphics_tools/gimp_advanced_guide/gimp_guide_node55.html
    dt = dst.dtype
    src = src.astype(n.single)
    dst = dst.astype(n.single)
    out = n.empty(src.shape, dtype = 'float')
    alpha = n.index_exp[:, :, 1]
    rgb = n.index_exp[:, :, 0]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = n.seterr(invalid = 'ignore')
    src_pre = src[rgb]*src_a
    dst_pre = dst[rgb]*dst_a
    # blend:
    blendfuncs = {
        MJBLEND_NORMAL: lambda s, d, sa_: s + d*sa_,
        MJBLEND_ADD: lambda s, d, sa_: n.minimum(255, s + d),
        MJBLEND_SUB: lambda s, d, sa_: n.maximum(0, s - d),
        MJBLEND_MULT: lambda s, d, sa_: s*d*sa_ / 255.0,
        MJBLEND_MULTINV: lambda s, d, sa_: (255.0 - s)*d*sa_ / 255.0,
        MJBLEND_SCREEN: lambda s, d, sa_: 255 - (1.0/255.0)*(255.0 - s)*(255.0 - d*sa_),
        MJBLEND_DIVIDE: lambda s, d, sa_: n.minimum(255, d*sa_*256.0 / (s + 1.0)),
        MJBLEND_MIN: lambda s, d, sa_: n.minimum(d*sa_, s),
        MJBLEND_MAX: lambda s, d, sa_: n.maximum(d*sa_, s),
    }
    out[rgb] = blendfuncs[blend_mode](src_pre, dst_pre, (1-src_a))
    out[rgb] /= out[alpha]
    n.seterr(**old_setting)
    out[alpha] *= 255
    n.clip(out,0,255)
    # astype('uint8') maps n.nan (and n.inf) to 0
    out = out.astype(dt)
    return out


class WordRenderer(object):

    def __init__(self, corpus, fontstate, colourstate, fillimstate, sz=(800,200)):
        # load corpus
        self.corpus = corpus() if isinstance(corpus,type) else corpus

        pygame.init()
        self.sz = sz
        self.screen = None

        # font render
        self.fontstate = fontstate() if isinstance(fontstate,type) else fontstate
        self.baselinestate = BaselineState()
        # border render
        self.borderstate = BorderState()
        # coloring
        self.colourstate = colourstate() if isinstance(colourstate,type) else colourstate
        self.fillimstate = fillimstate() if isinstance(fillimstate,type) else fillimstate
        # Projective distortion
        self.perspectivestate = PerspectiveTransformState()
        self.affinestate = AffineTransformState()
        self.diststate = DistortionState()
        self.surfdiststate = SurfaceDistortionState()
        self.elasticstate = ElasticDistortionState()

    def apply_perspective_surf(self, surf):
        invert_surface(surf)
        data = pygame.image.tostring(surf, 'RGBA')
        img = Image.fromstring('RGBA', surf.get_size(), data)
        img = img.transform(img.size, self.affinestate.proj_type,
            self.affinestate.sample_transformation(img.size),
            Image.BICUBIC)
        img = img.transform(img.size, self.perspectivestate.proj_type,
            self.perspectivestate.sample_transformation(img.size),
            Image.BICUBIC)
        im = n.array(img)
        # pyplot.imshow(im)
        # pyplot.show()
        surf = pygame.surfarray.make_surface(im[...,0:3].swapaxes(0,1))
        invert_surface(surf)
        return surf

    def apply_perspective_arr(self, arr, affstate, perstate, filtering=Image.BICUBIC):
        img = Image.fromarray(arr)
        img = img.transform(img.size, self.affinestate.proj_type,
            affstate,
            filtering)
        img = img.transform(img.size, self.perspectivestate.proj_type,
            perstate,
            filtering)
        arr = n.array(img)
        return arr

    def apply_perspective_rectim(self, rects, arr, affstate, perstate):
        rectarr = n.zeros(arr.shape)
        for i, rect in enumerate(rects):
            starti = max(0, rect[1])
            endi = min(rect[1]+rect[3], rectarr.shape[0])
            startj = max(0, rect[0])
            endj = min(rect[0]+rect[2], rectarr.shape[1])
            rectarr[starti:endi, startj:endj] = (i+1)*10
        rectarr = self.apply_perspective_arr(rectarr, affstate, perstate, filtering=Image.NONE)
        newrects = []
        for i, _ in enumerate(rects):
            try:
                newrects.append(pygame.Rect(get_bb(rectarr, eq=(i+1)*10)))
            except ValueError:
                pass
        return newrects

    def get_image(self):
        data = pygame.image.tostring(self.screen, 'RGBA')
        return n.array(Image.fromstring('RGBA', self.screen.get_size(), data))

    def get_ga_image(self, surf):
        r = pygame.surfarray.pixels_red(surf)
        a = pygame.surfarray.pixels_alpha(surf)
        r = r.reshape((r.shape[0], r.shape[1], 1))
        a = a.reshape(r.shape)
        return n.concatenate((r,a), axis=2).swapaxes(0,1)

    def get_bordershadow(self, bg_arr, colour):
        """
        Gets a border/shadow with the movement state [top, right, bottom, left].
        Inset or outset is random.
        """
        bs = self.borderstate.get_sample()
        outset = bs['outset']
        width = bs['width']
        position = bs['position']

        # make a copy
        border_arr = bg_arr.copy()
        # re-colour
        border_arr[...,0] = colour
        if outset:
            # dilate black (erode white)
            border_arr[...,1] = ndimage.grey_dilation(border_arr[...,1], size=(width, width))
            border_arr = arr_scroll(border_arr, position[0], position[1])

            # canvas = 255*n.ones(bg_arr.shape)
            # canvas = grey_blit(border_arr, canvas)
            # canvas = grey_blit(bg_arr, canvas)
            # pyplot.imshow(canvas[...,0], cmap=cm.Greys_r)
            # pyplot.show()

            return border_arr, bg_arr
        else:
            # erode black (dilate white)
            border_arr[...,1] = ndimage.grey_erosion(border_arr[...,1], size=(width, width))
            return bg_arr, border_arr

    def add_colour(self, canvas, fg_surf, border_surf=None):
        cs = self.colourstate.get_sample(2 + (border_surf is not None))
        # replace background
        pygame.PixelArray(canvas).replace((255,255,255), (cs[0],cs[0],cs[0]), distance=1.0)
        # replace foreground
        pygame.PixelArray(fg_surf).replace((0,0,0), (cs[1],cs[1],cs[1]), distance=0.99)

    def add_fillimage(self, arr):
        """
        Adds a fill image to the array.
        For blending this might be useful:
        - http://stackoverflow.com/questions/601776/what-do-the-blend-modes-in-pygame-mean
        - http://stackoverflow.com/questions/5605174/python-pil-function-to-divide-blend-two-images
        """
        # fis = self.fillimstate.get_sample(arr)

        # image = fis['image']
        # blend_mode = fis['blend_mode']
        # blend_amount = fis['blend_amount']
        # blend_order = fis['blend_order']

        # change alpha of the image
        # if blend_amount > 0:
        #     if blend_order:
        #         #image[...,1] *= blend_amount
        #         image[...,1] = (image[...,1]*blend_amount).astype(int)
        #         arr = grey_blit(image, arr, blend_mode=blend_mode)
        #     else:
        #         #arr[...,1] *= (1 - blend_amount)
        #         arr[...,1] = (arr[...,1]*(1-blend_amount)).astype(int)
        #         arr = grey_blit(arr, image, blend_mode=blend_mode)

        # pyplot.imshow(image[...,0], cmap=cm.Greys_r)
        # pyplot.show()

        return arr

    def mean_val(self, arr):
        return n.mean(arr[arr[...,1] > 0, 0].flatten())

    def surface_distortions(self, arr):
        ds = self.surfdiststate.get_sample()
        blur = ds['blur']

        origarr = arr.copy()
        arr = n.minimum(n.maximum(0, arr + n.random.normal(0, ds['noise'], arr.shape)), 255)
        # make some changes to the alpha
        arr[...,1] = ndimage.gaussian_filter(arr[...,1], ds['blur'])
        ds = self.surfdiststate.get_sample()
        arr[...,0] = ndimage.gaussian_filter(arr[...,0], ds['blur'])
        if ds['sharpen']:
            newarr_ = ndimage.gaussian_filter(origarr[...,0], blur/2)
            arr[...,0] = arr[...,0] + ds['sharpen_amount']*(arr[...,0] - newarr_)

        return arr

    def global_distortions(self, arr):
        # http://scipy-lectures.github.io/advanced/image_processing/#image-filtering
        ds = self.diststate.get_sample()

        blur = ds['blur']
        sharpen = ds['sharpen']
        sharpen_amount = ds['sharpen_amount']
        noise = ds['noise']

        newarr = n.minimum(n.maximum(0, arr + n.random.normal(0, noise, arr.shape)), 255)
        if blur > 0.1:
            newarr = ndimage.gaussian_filter(newarr, blur)
        if sharpen:
            newarr_ = ndimage.gaussian_filter(arr, blur/2)
            newarr = newarr + sharpen_amount*(newarr - newarr_)

        if ds['resample']:
            sh = newarr.shape[0]
            newarr = resize_image(newarr, newh=ds['resample_height'])
            newarr = resize_image(newarr, newh=sh)

        return newarr

    def apply_distortion_maps(self, arr, dispx, dispy):
        """
        Applies distortion maps generated from ElasticDistortionState
        """
        origarr = arr.copy()
        xx, yy = n.mgrid[0:dispx.shape[0], 0:dispx.shape[1]]
        xx = xx + dispx
        yy = yy + dispy
        coords = n.vstack([xx.flatten(), yy.flatten()])
        arr = ndimage.map_coordinates(origarr, coords, order=1, mode='nearest')
        return arr.reshape(origarr.shape)

    def generate_sample(self, display_text=None, display_text_length=None, outheight=None, pygame_display=False, random_crop=False, substring_crop=-1, char_annotations=False):
        """
        This generates the full text sample
        """
        if self.screen is None and pygame_display:
            self.screen = pygame.display.set_mode(self.sz)
            pygame.display.set_caption('WordRenderer')

        if display_text is None:
            # use random func to get display text and label
            display_text, label = self.corpus.get_sample(length=display_text_length)
        else:
            label = 0

        # get a new font state
        fs = self.fontstate.get_sample()
        # clear bg
        # bg_surf = pygame.Surface(self.sz, SRCALPHA, 32)
        bg_surf = pygame.Surface((round(2.0 * fs['size'] * len(display_text)), self.sz[1]), SRCALPHA, 32)

        font = freetype.Font(fs['font'], size=fs['size'])
        # random params
        display_text = fs['capsmode'](display_text) if fs['random_caps'] else display_text
        font.underline = fs['underline']
        font.underline_adjustment = fs['underline_adjustment']
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']
        font.antialiased = True
        char_spacing = fs['char_spacing']

        mid_idx = int(math.floor(len(display_text)/2))
        curve = [0 for c in display_text]
        rotations = [0 for c in display_text]

        if fs['curved'] and len(display_text) > 1:
            bs = self.baselinestate.get_sample()
            for i, c in enumerate(display_text[mid_idx+1:]):
                curve[mid_idx+i+1] = bs['curve'](i+1)
                rotations[mid_idx+i+1] = -int(math.degrees(math.atan(bs['diff'](i+1)/float(fs['size']/2))))
            for i,c in enumerate(reversed(display_text[:mid_idx])):
                curve[mid_idx-i-1] = bs['curve'](-i-1)
                rotations[mid_idx-i-1] = -int(math.degrees(math.atan(bs['diff'](-i-1)/float(fs['size']/2))))
            mean_curve = sum(curve) / float(len(curve)-1)
            curve[mid_idx] = -1*mean_curve

        # render text (centered)
        char_bbs = []
        # place middle char
        rect = font.get_rect(display_text[mid_idx])
        rect.centerx = bg_surf.get_rect().centerx
        rect.centery = bg_surf.get_rect().centery
        rect.centery +=  curve[mid_idx]
        bbrect = font.render_to(bg_surf, rect, display_text[mid_idx], rotation=rotations[mid_idx])
        bbrect.x = rect.x
        bbrect.y = rect.y
        char_bbs.append(bbrect)
        # render chars to the right
        last_rect = rect
        for i, c in enumerate(display_text[mid_idx+1:]):
            char_fact = 1.0
            if fs['random_kerning']:
                char_fact += fs['random_kerning_amount']*n.random.randn()
            newrect = font.get_rect(c)
            newrect.y = last_rect.y
            newrect.topleft = (last_rect.topright[0] + char_spacing*char_fact, newrect.topleft[1])
            newrect.centery = max(0 + newrect.height*1, min(self.sz[1] - newrect.height*1, newrect.centery + curve[mid_idx+i+1]))
            try:
                bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx+i+1])
            except ValueError:
                bbrect = font.render_to(bg_surf, newrect, c)
            bbrect.x = newrect.x
            bbrect.y = newrect.y
            char_bbs.append(bbrect)
            last_rect = newrect
        # render chars to the left
        last_rect = rect
        for i, c in enumerate(reversed(display_text[:mid_idx])):
            char_fact = 1.0
            if fs['random_kerning']:
                char_fact += fs['random_kerning_amount']*n.random.randn()
            newrect = font.get_rect(c)
            newrect.y = last_rect.y
            newrect.topright = (last_rect.topleft[0] - char_spacing*char_fact, newrect.topleft[1])
            newrect.centery = max(0 + newrect.height*1, min(self.sz[1] - newrect.height*1, newrect.centery + curve[mid_idx-i-1]))
            try:
                bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx-i-1])
            except ValueError:
                bbrect = font.render_to(bg_surf, newrect, c)
            bbrect.x = newrect.x
            bbrect.y = newrect.y
            char_bbs.append(bbrect)
            last_rect = newrect

        # Debug: show
        # self.screen = pygame.display.set_mode(bg_surf.get_size())
        # self.screen.fill((255,255,255))
        # self.screen.blit(bg_surf, (0,0))
        # for bb in char_bbs:
        #     pygame.draw.rect(self.screen, (255,0,0), bb, 2)
        # pygame.display.flip()
        # save_screen_img(self.screen, '00.jpg')
        # import pdb; pdb.set_trace()

        # border/shadow
        bg_arr = self.get_ga_image(bg_surf)
        # colour state
        cs = self.colourstate.get_sample(2 + fs['border'])
        # colour text
        bg_arr[...,0] = cs[0]
        if fs['border']:
            l1_arr, l2_arr = self.get_bordershadow(bg_arr, cs[2])
        else:
            l1_arr = bg_arr

        # Debug: show individual layers (fore, bord, back)
        # self.screen = pygame.display.set_mode(bg_surf.get_size())
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # globalcanvas = grey_blit(l2_arr, canvas)[...,0]
        # rgb_canvas = stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (0,0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '00.jpg')
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # globalcanvas = grey_blit(l1_arr, canvas)[...,0]
        # rgb_canvas = stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (50,50))
        # pygame.display.flip()
        # save_screen_img(self.screen, '00.jpg')
        # self.screen.fill((cs[1],cs[1],cs[1]))
        # pygame.display.flip()
        # save_screen_img(self.screen, '01.jpg')
        # import pdb; pdb.set_trace()


        # Projective distortion
        affstate = self.affinestate.sample_transformation(l1_arr.shape)
        perstate = self.perspectivestate.sample_transformation(l1_arr.shape)
        l1_arr[...,1] = self.apply_perspective_arr(l1_arr[...,1], affstate, perstate)

        if fs['border']:
            l2_arr[...,1] = self.apply_perspective_arr(l2_arr[...,1], affstate, perstate)
        if char_annotations:
            char_bbs = self.apply_perspective_rectim(char_bbs, l1_arr[...,1], affstate, perstate)
            # order char_bbs by left to right
            xvals = [bb.x for bb in char_bbs]
            idx = [i[0] for i in sorted(enumerate(xvals), key=lambda x:x[1])]
            char_bbs = [char_bbs[i] for i in idx]

        if n.random.rand() < substring_crop and len(display_text) > 4 and char_annotations:
            # randomly crop to just a sub-string of the word
            start = n.random.randint(0, len(display_text)-1)
            stop = n.random.randint(min(start+1,len(display_text)), len(display_text))
            display_text = display_text[start:stop]
            char_bbs = char_bbs[start:stop]
            # get new bb of image
            bb = pygame.Rect(get_rects_union_bb(char_bbs, l1_arr))
        else:
            # get bb of text
            if fs['border']:
                bb = pygame.Rect(get_bb(grey_blit(l2_arr, l1_arr)[...,1]))
            else:
                bb = pygame.Rect(get_bb(l1_arr[...,1]))
        if random_crop:
            bb.inflate_ip(10*n.random.randn()+15, 10*n.random.randn()+15)
        else:
            inflate_amount = int(0.4*bb[3])
            bb.inflate_ip(inflate_amount, inflate_amount)


        # crop image
        l1_arr = imcrop(l1_arr, bb)
        if fs['border']:
            l2_arr = imcrop(l2_arr, bb)
        if char_annotations:
            # adjust char bbs
            for char_bb in char_bbs:
                char_bb.move_ip(-bb.x, -bb.y)
        canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        canvas[...,0] = cs[1]

        # Debug: show
        # globalcanvas = grey_blit(l1_arr, canvas)
        # if fs['border']:
        #     globalcanvas = grey_blit(l2_arr, globalcanvas)
        # globalcanvas = globalcanvas[...,0]
        # rgb_canvas = stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1))
        # for char_bb in char_bbs:
        #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
        # self.screen = pygame.display.set_mode(canvas_surf.get_size())
        # self.screen.blit(canvas_surf, (0, 0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '01.jpg')
        # import pdb; pdb.set_trace()


        # Natural data blending
        try:
            canvas = self.add_fillimage(canvas)
            l1_arr = self.add_fillimage(l1_arr)
            if fs['border']:
                l2_arr = self.add_fillimage(l2_arr)
        except Exception:
            print("\tfillimage error")
            return None

        # add per-surface distortions
        import pdb; pdb.set_trace()

        l1_arr = self.surface_distortions(l1_arr)
        if fs['border']:
            l2_arr = self.surface_distortions(l2_arr)

        # compose global image
        blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
        count = 0
        while True:
            globalcanvas = grey_blit(l1_arr, canvas, blend_mode=random.choice(blend_modes))
            if fs['border']:
                globalcanvas = grey_blit(l2_arr, globalcanvas, blend_mode=random.choice(blend_modes))
            globalcanvas = globalcanvas[...,0]
            std = n.std(globalcanvas.flatten())
            count += 1
            #print count
            if std > 20:
                break
            if count > 10:
                print("\tcan't get good contrast")
                return None

        canvas = globalcanvas

        # do elastic distortion described by http://research.microsoft.com/pubs/68920/icdar03.pdf
        # dispx, dispy = self.elasticstate.sample_transformation(canvas.shape)
        # canvas = self.apply_distortion_maps(canvas, dispx, dispy)

        # add global distortions
        canvas = self.global_distortions(canvas)

        cv2.imwrite("1.jpg", canvas)
        import pdb; pdb.set_trace()

        # noise removal
        canvas = ndimage.filters.median_filter(canvas, size=(3,3))

        # resize
        if outheight is not None:
            if char_annotations:
                char_bbs = resize_rects(char_bbs, canvas, outheight)
            canvas = resize_image(canvas, newh=outheight)

        # FINISHED, SHOW ME SOMETHING
        if pygame_display:
            rgb_canvas = self.stack_arr((canvas, canvas, canvas))
            canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1))
            # for char_bb in char_bbs:
            #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
            self.screen = pygame.display.set_mode(canvas_surf.get_size())
            self.screen.blit(canvas_surf, (0, 0))
            pygame.display.flip()

        return {
            'image': canvas,
            'text': display_text,
            'label': label,
            'chars': n.array([[c.x, c.y, c.width, c.height] for c in char_bbs])}