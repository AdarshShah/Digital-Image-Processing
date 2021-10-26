import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft2, ifft2,fftshift, ifftshift
from scipy.io import loadmat
from scipy.signal.signaltools import resample
from skimage.io import imread, imsave
from skimage.util.dtype import img_as_float, img_as_ubyte
from math import sqrt

#Here T is the Threshold to handle inverse issues.
def InverseFilter(img,T,file):

    #FFT  
    img = fft2(img)

    P,Q = np.shape(img)

    #Obtaining inverse filter
    kernel = loadmat('BlurKernel.mat')
    p,q = np.shape(kernel['h'])
    mask = np.pad(kernel['h'],((0,P-p),(0,Q-q)),mode='constant',constant_values=0)
    mask = fft2(mask)

    #Thresholding
    mask = np.where(np.absolute(mask)<T,mask*T/np.absolute(mask),mask)

    #Applying Filter
    img = img/mask

    #generating resultant image

    resimg = np.absolute(ifft2(img))
    #normalizing the float image
    mx,mn = np.max(resimg),np.min(resimg)
    resimg = (resimg-mn)/(mx-mn)
    
    resimg = img_as_ubyte(resimg)

    imsave('./outputs/Q2a_'+file,resimg)


#Here K is the tuning Parameter. K = inv(SNR)
def WeinerFilter(img,sigma,file):

    img = img_as_float(img)

    P,Q = np.shape(img)

    #load kernel
    kernel = loadmat('BlurKernel.mat')
    p,q = np.shape(kernel['h'])
    mask = np.pad(kernel['h'],((0,P-p),(0,Q-q)),mode='constant',constant_values=0)
    mask = fft2(mask)
    H = np.conjugate(fftshift(mask))

    #Weiner Filter in Frequency Domain
    inverse_SNR = np.fromfunction(lambda x,y: sigma**2 * np.sqrt((x-P//2)**2+(y-Q//2)**2)/10**5,(P,Q))

    D = H/(np.absolute(H)**2+inverse_SNR)

    #Image in frquency Domain
    G = fftshift(fft2(img))

    #Filtering
    F = D*G

    #generating resultant image
    freq = ifftshift(F)
    resimg = np.absolute(ifft2(freq))
    #normalizing the float image
    mx,mn = np.max(resimg),np.min(resimg)
    resimg = (resimg-mn)/(mx-mn)
    
    resimg = img_as_ubyte(resimg)

    imsave('./outputs/Q2b_'+file,resimg)


if __name__=='__main__':
    for file in ['Blurred_LowNoise.png','Blurred_HighNoise.png']:
        InverseFilter(imread(file),0.1,file)
    
    WeinerFilter(imread('Blurred_LowNoise.png'),1,'Blurred_LowNoise.png')
    WeinerFilter(imread('Blurred_HighNoise.png'),10,'Blurred_HighNoise.png')


