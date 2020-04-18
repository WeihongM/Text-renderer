import math
import numpy as n
from PIL import Image
from scipy import ndimage, interpolate

def matrix_mult(A, B):
    C = n.empty((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i,j] = n.sum(A[i,:]*B[:,j])
    return C

class AffineTransformState(object):
    """
    Defines the random state for an affine transformation
    """
    proj_type = Image.AFFINE
    rotation = [0, 1]  # rotate normal dist mean, std
    skew = [0, 0]  # skew normal dist mean, std

    def sample_transformation(self, imsz):
        theta = math.radians(self.rotation[1]*n.random.randn() + self.rotation[0])
        ca = math.cos(theta)
        sa = math.sin(theta)
        R = n.zeros((3,3))
        R[0,0] = ca
        R[0,1] = -sa
        R[1,0] = sa
        R[1,1] = ca
        R[2,2] = 1
        S = n.eye(3,3)
        S[0,1] = math.tan(math.radians(self.skew[1]*n.random.randn() + self.skew[0]))
        A = matrix_mult(R,S)
        x = imsz[1]/2
        y = imsz[0]/2
        return (A[0,0], A[0,1], -x*A[0,0] - y*A[0,1] + x,
            A[1,0], A[1,1], -x*A[1,0] - y*A[1,1] + y)


class PerspectiveTransformState(object):
    """
    Defines teh random state for a perspective transformation
    Might need to use http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    """
    proj_type = Image.PERSPECTIVE
    a_dist = [1, 0.01]
    b_dist = [0, 0.005]
    c_dist = [0, 0.005]
    d_dist = [1, 0.01]
    e_dist = [0, 0.0005]
    f_dist = [0, 0.0005]

    def v(self, dist):
        return dist[1]*n.random.randn() + dist[0]

    def sample_transformation(self, imsz):
        x = imsz[1]/2
        y = imsz[0]/2
        a = self.v(self.a_dist)
        b = self.v(self.b_dist)
        c = self.v(self.c_dist)
        d = self.v(self.d_dist)
        e = self.v(self.e_dist)
        f = self.v(self.f_dist)

        # scale a and d so scale kept same
        #a = 1 - e*x
        #d = 1 - f*y

        z = -e*x - f*y + 1
        A = n.zeros((3,3))
        A[0,0] = a + e*x
        A[0,1] = b+f*x
        A[0,2] = -a*x-b*y-e*x*x-f*x*y+x
        A[1,0] = c+e*y
        A[1,1] = d+f*y
        A[1,2] = -c*x-d*y-e*x*y-f*y*y+y
        A[2,0] = e
        A[2,1] = f
        A[2,2] = z
        # print(a,b,c,d,e,f)
        # print(z)
        A = A / z

        return (A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], A[2,0], A[2,1])


class ElasticDistortionState(object):
    """
    Defines a random state for elastic distortions
    """
    displacement_range = 1
    alpha_dist = [[15, 30], [0, 2]]
    sigma = [[8, 2], [0.2, 0.2]]
    min_sigma = [4, 0]

    def sample_transformation(self, imsz):
        choices = len(self.alpha_dist)
        c = int(n.random.randint(0, choices))
        sigma = max(self.min_sigma[c], n.abs(self.sigma[c][1]*n.random.randn() + self.sigma[c][0]))
        alpha = n.random.uniform(self.alpha_dist[c][0], self.alpha_dist[c][1])
        dispmapx = n.random.uniform(-1*self.displacement_range, self.displacement_range, size=imsz)
        dispmapy = n.random.uniform(-1*self.displacement_range, self.displacement_range, size=imsz)
        dispmapx = alpha * ndimage.gaussian_filter(dispmapx, sigma)
        dispmaxy = alpha * ndimage.gaussian_filter(dispmapy, sigma)
        return dispmapx, dispmaxy

class DistortionState(object):
    blur = [0, 1]
    sharpen = 0
    sharpen_amount = [30, 10]
    noise = 4
    resample = 0.1
    resample_range = [24, 32]

    def get_sample(self):
        return {
            'blur': n.abs(self.blur[1]*n.random.randn() + self.blur[0]),
            'sharpen': n.random.rand() < self.sharpen,
            'sharpen_amount': self.sharpen_amount[1]*n.random.randn() + self.sharpen_amount[0],
            'noise': self.noise,
            'resample': n.random.rand() < self.resample,
            'resample_height': int(n.random.uniform(self.resample_range[0], self.resample_range[1]))
        }

class SurfaceDistortionState(DistortionState):
    noise = 8
    resample = 0
