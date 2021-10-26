import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack.basic import fft2, ifft2
from scipy.io import loadmat
from scipy.signal.windows.windows import gaussian
from skimage.io import imread, imsave
from skimage.util.dtype import img_as_float, img_as_ubyte
from scipy.signal import convolve2d
from math import exp, pi


def MeanFilter(img,N):

    img = img_as_float(img)

    #Mean mask
    mask = np.ones((N,N))/(N*N)
    
    #Convolution with img
    resimg = convolve2d(img,mask,mode='same')

    return img_as_ubyte(resimg)


def MedianFilter(img,N):

    resimg = img.copy()
    P,Q = np.shape(img)

    #Median 
    for i in range(P):
        for j in range(Q):
            resimg[i][j] = np.median(img[max(0,i-N//2):min(P-1,i+N//2),max(0,j-N//2):min(Q-1,j+N//2)])
    
    return resimg

#Utility function to calculate gaussian
def gauss(x,sigma):
    g = exp(-(x/sigma)**2/2)
    return g

def BilateralFilter(img,sigma):

    img = img_as_float(img)

    P,Q = np.shape(img)

    #Resultant image
    resimg = np.zeros(np.shape(img))

    #Tuning Parameters

    #K : Square Window Size = 2*K*sigma X 2*K*sigma
    K=1

    #Ratio of standard deviation of Spatial Gaussian Filter and Noise Gaussian Filter
    #Used Trial and error to get better results
    beta=0.35

    for i in range(P):
        for j in range(Q):
            sum=0
            for k in range(-K*sigma,K*sigma+1,1):
                for l in range(-K*sigma,K*sigma+1,1):
                    try:
                        if (i+k>=0 and i+k<P) and (j+l>=0 and j+l<Q) and (i-k>=0 and i-k<P) and (j-l>=0 and j-l<Q):
                            g = gauss(k,sigma*beta)*gauss(l,sigma*beta)*gauss(img[i+k,j+l]-img[i,j],sigma)
                            resimg[i][j]+= img[i+k][j+l]*g
                            sum += g
                        else:
                            #Wrap around Boundary
                            g = gauss(k,sigma*beta)*gauss(l,sigma*beta)*gauss(img[i-k,j-l]-img[i,j],sigma)
                            resimg[i][j]+= img[i-k][j-l]*g
                            sum += g
                    except IndexError:
                        #Occurs at boundaries
                        pass
            resimg[i][j]/=g
    
    #Full Scale Constrast stretch for correction
    mx,mn = np.max(resimg),np.min(resimg)
    resimg = (resimg-mn)/(mx-mn)

    return img_as_ubyte(resimg)



if __name__=='__main__':
    
    img = imread('noisy_book1.png')
    
    mean_img = MeanFilter(img,5)
    imsave('./outputs/Q3a_mean.png',mean_img)

    median_img = MedianFilter(img,5)
    imsave('./outputs/Q3a_median.png',median_img)

    img = imread('noisy_book2.png')

    gauss_img = BilateralFilter(img,5)
    imsave('./outputs/Q3b_bilateral.png',gauss_img)









