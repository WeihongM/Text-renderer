import os, re
import random
import numpy as np

class Corpus(object):
    """
    Defines a corpus of words
    """
   # valid_ascii = [48,49,50,51,52,53,54,55,56,57,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90]
    valid_ascii = [36,37,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,61,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122]

    def __init__(self):
        pass


class TestCorpus(Corpus):
    """
    Just a test corpus from a text file
    """
    CORPUS_FN = "./corpus.txt"

    def __init__(self, args={'unk_probability': 0}):
        self.corpus_text = ""
        pattern = re.compile('[^a-zA-Z0-9 ]')
        for line in open(self.CORPUS_FN):
            line = line.replace('\n', ' ')
            line = pattern.sub('', line)
            self.corpus_text = self.corpus_text + line
        self.corpus_text = ''.join(c for c in self.corpus_text if c.isalnum() or c.isspace())
        self.corpus_list = self.corpus_text.split()
        self.unk_probability = args['unk_probability']

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = np.random.randint(0, len(self.corpus_list))
        breakamount = 1000
        count = 0
        while not sampled:
            samp = self.corpus_list[idx]
            if length > 0:
                if len(samp) >= length:
                    if len(samp) > length:
                        # start at a random point in this word
                        diff = len(samp) - length
                        starti = np.random.randint(0, diff)
                        samp = samp[starti:starti+length]
                    break
            else:
                break
            idx = np.random.randint(0, len(self.corpus_list))
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")
        if np.random.rand() < self.unk_probability:
            # change some letters to make it random
            ntries = 0
            while True:
                ntries += 1
                if len(samp) > 2:
                    n_to_change = np.random.randint(2, len(samp))
                else:
                    n_to_change = max(1, len(samp) - 1)
                idx_to_change = np.random.permutation(len(samp))[0:n_to_change]
                samp = list(samp)
                for idx in idx_to_change:
                    samp[idx] = chr(random.choice(self.valid_ascii))
                samp = "".join(samp)
                if samp not in self.corpus_list:
                    idx = len(self.corpus_list)
                    break
                if ntries > 10:
                    idx = self.corpus_list.index(samp)
                    break
        return samp, idx


class ChineseCorpus(Corpus):
    """
    Just a Chinese corpus from a text file
    """
    def __init__(self,corpus_folder,args={'unk_probability': 0}):
        self.corpus_text = []
        files = os.listdir(corpus_folder['corpus_folder'])
        for file in files:
            with open(os.path.join(corpus_folder['corpus_folder'],file),'r') as fp:
                for line in fp.readlines():
                    line=line.strip()
                    if (len(line) > 16 or len(line) == 0):
                        continue
                    self.corpus_text.append(line)
        random.shuffle(self.corpus_text)
        self.nSample = len(self.corpus_text)

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = np.random.randint(0,self.nSample)
        breakamount = 1000
        count = 0
        while not sampled:
            samp = self.corpus_text[idx]
            if len(samp) > 0:
                break
            idx = np.random.randint(0, self.nSample)
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")
        return samp, len(samp)


class EnglishLineCorpus(Corpus):
    """
    Just a Chinese corpus from a text file
    """
    def __init__(self,corpus_folder,args={'unk_probability': 0}):
        self.corpus_text = []
        files = os.listdir(corpus_folder['corpus_folder'])
        for file in files:
            with open(os.path.join(corpus_folder['corpus_folder'],file),'r') as fp:
                for line in fp.readlines():
                    line=line.strip()
                    line=line.decode('utf-8')
                    self.corpus_text.append(line)
        random.shuffle(self.corpus_text)
        self.nSample = len(self.corpus_text)

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = np.random.randint(0,self.nSample)
        breakamount = 1000
        count = 0
        nword = random.randint(2,3)
        
        while not sampled:
            samp = self.corpus_text[idx]
            if len(samp) > 0:
                words = samp.split()
                dw = len(words) - nword
                if dw > 0:
                    first_word_index = random.choice(range(dw+1))
                    samp = ' '.join(words[first_word_index:first_word_index+nword])

                    if len(samp) < 20:
                        break
            idx = np.random.randint(0, self.nSample)
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")
        return samp, len(samp)


class SVTCorpus(TestCorpus):
    CORPUS_FN = "/Users/jaderberg/Data/TextSpotting/DataDump/svt1/svt_lex_lower.txt"


class FileCorpus(TestCorpus):
    def __init__(self, args):
        self.CORPUS_FN = args['fn']
        TestCorpus.__init__(self, args)


class NgramCorpus(TestCorpus):
    """
    Spits out a word sample, dictionary label, and ngram encoding labels
    """
    def __init__(self, args):
        words_fn = args['encoding_fn_base'] + '_words.txt'
        idx_fn = args['encoding_fn_base'] + '_idx.txt'
        values_fn = args['encoding_fn_base'] + '_values.txt'

        self.words = self._load_list(words_fn)
        self.idx = self._load_list(idx_fn, split=' ', tp=int)
        self.values = self._load_list(values_fn, split=' ', tp=int)

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = np.random.randint(0, len(self.words))
        breakamount = 1000
        count = 0
        while not sampled:
            samp = self.words[idx]
            if length > 0:
                if len(samp) >= length:
                    if len(samp) > length:
                        # start at a random point in this word
                        diff = len(samp) - length
                        starti = np.random.randint(0, diff)
                        samp = samp[starti:starti+length]
                    break
            else:
                break
            idx = np.random.randint(0, len(self.words))
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")

        return samp, {
            'word_label': idx,
            'ngram_labels': self.idx[idx],
            'ngram_counts': self.values[idx],
        }

    def _load_list(self, listfn, split=None, tp=str):
        arr = []
        for l in open(listfn):
            l = l.strip()
            if split is not None:
                l = [tp(x) for x in l.split(split)]
            else:
                l = tp(l)
            arr.append(l)
        return arr


class RandomCorpus(Corpus):
    """
    Generates random strings
    """
    def __init__(self, args={'min_length': 1, 'max_length': 23}):
        self.min_length = args['min_length']
        self.max_length = args['max_length']

    def get_sample(self, length=None):
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        samp = ""
        for i in range(length):
            samp = samp + chr(random.choice(self.valid_ascii))
        return samp, length