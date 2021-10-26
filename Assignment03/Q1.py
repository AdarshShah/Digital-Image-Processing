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

def q1a():
    #Parameters
    M,N = 1001,1001
    u0,v0 = 100,200
    #Generate Image using vectorization
    img = np.fromfunction(function=lambda i,j: np.sin(2*np.pi*u0*i/M + 2*np.pi*v0*j/N),shape=(M,N))
    #Generate FFT 
    result = np.log2(np.absolute(fft2(img)))
    mn,mx = np.min(result),np.max(result)
    #Full Scale Contrast Stretch for normalization
    result = (result-mn)/(mx-mn)
    result = img_as_ubyte(result)
    imsave('outputs/Q1a.jpeg',result)


def q1b(D0):
    #Read image 
    img = imread('./characters.tif')
    img = img_as_float(img)
    #obtain fft
    freq = fft2(img)
    #shift origin to center
    freq = fftshift(freq)
    #Rows and columns
    P,Q = img.shape
    #Generating Ideal low pass filter Mask
    m = np.fromfunction(function=lambda x,y: (x-P//2)**2+(y-Q//2)**2<=D0**2,shape=(P,Q))
    
    #apply filter
    freq = freq*m

    #generating resultant image
    freq = ifftshift(freq)
    resimg = np.absolute(ifft2(freq))
    #normalizing the float image
    mx,mn = np.max(resimg),np.min(resimg)
    resimg = (resimg-mn)/(mx-mn)
    #saving the result image
    resimg = img_as_ubyte(resimg)
    imsave('./outputs/Q1b.jpeg',resimg)

def q1c(D0):
    #Read image 
    img = imread('./characters.tif')
    img = img_as_float(img)
    #obtain fft
    freq = fft2(img)
    #shift origin to center
    freq = fftshift(freq)
    #Rows and columns
    P,Q = img.shape
    #Generating Gaussian Mask
    m = np.fromfunction(function=lambda x,y: np.exp(-((x-P//2)**2+(y-Q//2)**2)/(2*D0**2)),shape=(P,Q))
    #apply filter
    freq = freq*m
    #generating resultant image
    freq = ifftshift(freq)
    resimg = np.absolute(ifft2(freq))
    #normalizing the float image
    mx,mn = np.max(resimg),np.min(resimg)
    resimg = (resimg-mn)/(mx-mn)
    #saving the result image
    resimg = img_as_ubyte(resimg)
    imsave('./outputs/Q1c.jpeg',resimg)


if __name__=='__main__':
    q1a()
    q1b(D0=100)
    q1c(D0=100)