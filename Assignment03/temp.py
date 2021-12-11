import matplotlib
from numpy.core.fromnumeric import shape
from numpy.fft.helper import ifftshift
from scipy.fftpack.basic import fft2, ifft2
from scipy.fftpack import fftshift
import skimage, cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imsave,imshow,imread
from skimage.util.dtype import img_as_float, img_as_ubyte
from math import exp, sqrt
from scipy.signal import convolve2d

if __name__=='__main__':
    # img = imread('Original_book.png')

    # L = np.zeros((3,3))
    # L[0][1] = L[1][2] = L[2][1] = L[1][0] = -1
    # L[1][1] = 4

    # result = convolve2d(img,L,'same')

    # x = np.mean(result)
    
    # result = np.where(result<x,0,255)
    # plt.imshow(result,'gray')
    # plt.show()

    #Ideal Low Pass Filter
    H = np.zeros(11)
    H[3:8]=1
    plt.subplot(121)
    plt.scatter(np.arange(len(H)),H)

    plt.subplot(122)
    h = np.fft.fftshift(np.fft.ifft(H))
    plt.scatter(np.arange(len(h)),h)
    plt.show()