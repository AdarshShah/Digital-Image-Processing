from math import radians
from numpy import linalg
from numpy.linalg.linalg import det
from skimage.io import imread, imsave
from skimage import img_as_float,img_as_ubyte
from skimage.filters.edges import sobel, laplacian, sobel_h, sobel_v
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity
from skimage.transform import rotate,resize
from skimage.filters import gaussian
from skimage.util import random_noise
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np



def HarrisCornerDetector(img,N):

    #Essential Conversions to grayscale and float
    img = img_as_float(img)

    #Extract Edges using laplacian
    Ix = sobel_h(img)
    Iy = sobel_v(img)

    #Ix^2
    A = np.square(Ix)
    #IxIy
    B = Ix*Iy
    #Iy^2
    C = np.square(Iy)

    #Vectorized Implementation
    Mask = np.ones((N,N))    

    A = convolve2d(A,Mask,'same')
    B = convolve2d(B,Mask,'same')
    C = convolve2d(C,Mask,'same')

    result = A*C-B**2-0.06*(A+C)

    result = rescale_intensity(result,out_range=(0,1))
    return img_as_ubyte(result)

if __name__=='__main__':

    img = imread('Checkerboard.png')
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_Checkerboard.png',result)

    img = imread('Checkerboard.png')
    img = rotate(img,45)
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_Checkerboard_rotated.png',result)

    img = imread('Checkerboard.png')
    img = resize(img,(100,100))
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_Checkerboard_rescaled.png',result)

    img = imread('Checkerboard.png')
    img = random_noise(img,'gaussian')
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_Checkerboard_noisy.png',result)

    img = imread('MainBuilding.png')
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_MainBuilding.png',result)

    img = imread('MainBuilding.png')
    img = rotate(img,45)
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_MainBuilding_rotated.png',result)

    img = imread('MainBuilding.png')
    img = resize(img,(100,100))
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_MainBuilding_rescaled.png',result)

    img = imread('MainBuilding.png')
    img = random_noise(img,'gaussian')
    result = HarrisCornerDetector(img=img,N=10)
    imsave('Harris_MainBuilding_noisy.png',result)
