import numpy as n
import random

class FontState(object):
    """
    Defines the random state of the font rendering
    """
    size = [60, 10]  # normal dist mean, std
    underline = 0.
    strong = 0.5
    oblique = 0.001
    wide = 0.5
    strength = [0.02778, 0.05333]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.1
    random_caps = 1.0
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 0.
    random_kerning = 0.
    random_kerning_amount = 0.1

    def __init__(self, font_list):
        self.FONT_LIST = font_list
        self.fonts = [f.strip() for f in open(self.FONT_LIST)]

    def get_sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(n.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*n.random.randn() + self.size[0],
            'underline': n.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1]*n.random.randn() + self.underline_adjustment[0])),
            'strong': n.random.rand() < self.strong,
            'oblique': n.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0])*n.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3]*(n.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': n.random.rand() < self.border,
            'random_caps': n.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': n.random.rand() < self.curved,
            'random_kerning': n.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }

class BaselineState(object):
    curve = lambda this, a: lambda x: a*x*x
    differential = lambda this, a: lambda x: 2*a*x
    a = [0, 0.1]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        a = self.a[1]*n.random.randn() + self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }