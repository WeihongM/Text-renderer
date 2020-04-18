import numpy as n

class BorderState(object):
    outset = 0.5
    width = [4, 4]  # normal dist
    position = [[0,0], [-1,-1], [-1,1], [1,1], [1,-1]]

    def get_sample(self):
        p = self.position[int(n.random.randint(0,len(self.position)))]
        w = max(1, int(self.width[1]*n.random.randn() + self.width[0]))
        return {
            'outset': n.random.rand() < self.outset,
            'width': w,
            'position': [int(-1*n.random.uniform(0,w*p[0]/1.5)), int(-1*n.random.uniform(0,w*p[1]/1.5))]
        }