from skimage.io import imread, imsave
from skimage import img_as_float,img_as_ubyte
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np
from cv2 import pyrDown

def gaussian(img:np.ndarray,sigma:int,n:int):

    img = img_as_float(img)

    h = np.fromfunction(function=lambda x,y : np.exp(-((x-n//2)**2+(y-n//2)**2)/(2*sigma**2)),shape=(n,n))
    h = h/np.sum(h)

    result = convolve2d(img,h,'same')

    return img_as_ubyte(result)

def downsample(img:np.ndarray,k:int):

    P,Q = np.shape(img)

    result = np.zeros((P//k,Q//k),dtype=np.ubyte)
    
    for i in range(P//k):
        for j in range(Q//k):
            result[i][j] = img[k*i][k*j]

    return result

if __name__=='__main__':

    img = imread('barbara.tif')

    downsample_img = downsample(img,2)
    imsave('Q1_downsample_without_gaussian.png',downsample_img)

    gaussian_img = gaussian(img,1,5)
    downsample_img = downsample(gaussian_img,2)
    imsave('Q1_downsample_with_gaussian.png',downsample_img)


    inbuilt_img = pyrDown(img)
    imsave('Q1_inbuiltdownsample.png',inbuilt_img)

