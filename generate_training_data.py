import sys, os
import numpy as np
import cv2, time
import os.path as osp

from PIL import Image
from word_render import WordRenderer
from lib.border_render import BorderState
from lib.color import (
    ColourState,
    TrainingCharsColourState,
)
from lib.data_blend import (
    FillImageState,
    SVTFillImageState,
)
from lib.font_render import (
        FontState,
        BaselineState,
)
from lib.corpus import (
    Corpus,
    RandomCorpus,
    ChineseCorpus,
    FileCorpus
)


SETTINGS = {
    "chinese":{
        'corpus_class': ChineseCorpus,
        'corpus_args': {'corpus_folder': 'sample/data2/chinese', 'min_length': 1, 'max_length': 10},
        'fontstate':{
            'font_list': "sample/data2/font_path_list.txt",
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
                    },
        'trainingchars_fn': "sample/data/word_img_path.txt",
        'fillimstate': {
            'gtmat_fn': "sample/data2/background.txt"
                        },
                },
    'RAND10': {
        'corpus_class': RandomCorpus,
        'corpus_args': {'min_length': 1, 'max_length': 10},
        'fontstate':{
            'font_list': ["sample/data/font_path_list.txt",
                        "sample/data/font_path_list.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
                    },
        'trainingchars_fn': ["sample/data/word_img_path.txt",
                            "sample/data/word_img_path.txt"],
        'fillimstate': {
            'gtmat_fn':["sample/data/svt_img_path.txt",
                        "sample/data/svt_img_path.txt"]
                        }
                },
        }


def create_synthetic_data():
    num_to_generate = 5
    save_dir = "vis"
    QUALITY = [80, 10]
    SAMPLE_HEIGHT = 100

    dataset = "chinese"
    settings = SETTINGS[dataset]

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    sz = (1800,300)
    # init providers
    if 'corpus_class' in settings:
        corp_class = settings['corpus_class']
    else:
        corp_class = FileCorpus
    if 'corpus_args' in settings:
        corpus = corp_class(settings['corpus_args'])
    else:
        corpus = corp_class()
    # init fontstate
    fontstate = FontState(font_list=settings['fontstate']['font_list'])
    fontstate.random_caps = settings['fontstate']['random_caps']
    # init colourstate
    colourstate = TrainingCharsColourState(settings['trainingchars_fn'])
    # init fillimstate
    if not isinstance(settings['fillimstate'], list):
        fillimstate = SVTFillImageState(settings['fillimstate']['gtmat_fn'])
    else:
        # its a list of different fillimstates to combine
        states = []
        for i, fs in enumerate(settings['fillimstate']):
            s = SVTFillImageState(fs['gtmat_fn'])
            # move datadir to imlist
            s.IMLIST = [l for l in s.IMLIST]
            states.append(s)
        fillimstate = states.pop()
        for fs in states:
            fillimstate.IMLIST.extend(fs.IMLIST)

    # init substrings
    try:
        substr_crop = settings['substrings']
    except KeyError:
        substr_crop = -1

    WR = WordRenderer(corpus=corpus, fontstate=fontstate, colourstate=colourstate, fillimstate=fillimstate, sz=sz)
    count = 0

    for i in range(num_to_generate):
        data = WR.generate_sample(outheight=SAMPLE_HEIGHT, random_crop=True, substring_crop=substr_crop, char_annotations=True)
        if data is None:
            print("\tcould not generate good sample")
            continue
        try:
            img = Image.fromarray(data['image'])
        except Exception:
            print("\tbad image generated")
            continue

        if img.mode != 'RGB':
            img = img.convert('RGB')
        quality = min(80, max(0, int(QUALITY[1]*np.random.randn() + QUALITY[0])) )
        img.save(osp.join(save_dir+"/"+str(count)+'.jpg'),'JPEG',quality=quality)
        print("count: ", count, "image label: ", data['text'])
        count += 1


if __name__ == '__main__':
    create_synthetic_data()