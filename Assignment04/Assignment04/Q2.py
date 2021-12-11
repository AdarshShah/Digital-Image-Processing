from skimage.io import imread, imsave
from skimage import img_as_float,img_as_ubyte
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np

def gaussian(img:np.ndarray,sigma:int,n:int):

    img = img_as_float(img)

    h = np.fromfunction(function=lambda x,y : np.exp(-((x-n//2)**2+(y-n//2)**2)/(2*sigma**2)),shape=(n,n))
    h = h/np.sum(h)

    result = convolve2d(img,h,'same')

    return img_as_ubyte(result)

def sobelPrewitt(img,threshold):

    img = img_as_float(img)

    sx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sy = [[1,2,1],[0,0,0],[-1,-2,-1]]

    img_sx = convolve2d(img,sx,'same')
    img_sy = convolve2d(img,sy,'same')

    result = np.sqrt(img_sx**2 + img_sy**2)

    mx,mn = np.max(result),np.min(result)
    result = (result-mn)/(mx-mn)

    result = np.where(result>threshold,255,0)

    return img_as_ubyte(result)


def laplacian(img,threshold):

    img = img_as_float(img)

    L = [[0,1,0],[1,-4,1],[0,1,0]]

    result = convolve2d(img,L,'same')

    mx,mn = np.max(result),np.min(result)
    result = 2*(result-mn)/(mx-mn)-1

    result = np.where(np.absolute(result)>threshold,np.sign(result)*1,0)

    H1 = np.zeros((5,5))
    H2 = np.zeros((5,5))

    H1[2,:2],H1[2,3:]=1,-1
    H2[:2,2],H2[3:,2]=1,-1

    l1 = convolve2d(result,H1,'same')
    l2 = convolve2d(result,H2,'same')
    
    l1 = np.absolute(l1)
    l2 = np.absolute(l2)

    laplace = (l1+l2)/2

    laplace = np.where(laplace>0.1,255,0)

    return img_as_ubyte(laplace)


if __name__=='__main__':

    file = 'Checkerboard.png'
    img = imread(file)
    gaussian_img = gaussian(img,5,5)
    sobelPrewitt_img = sobelPrewitt(gaussian_img,threshold=0.46)
    laplacian_img = laplacian(gaussian_img,threshold=0.25)
    imsave('laplacian_'+file,laplacian_img)
    imsave('sobelPrewitt_'+file,sobelPrewitt_img)

    file = 'NoisyCheckerboard.png'
    img = imread(file)
    gaussian_img = gaussian(img,5,5)
    sobelPrewitt_img = sobelPrewitt(gaussian_img,threshold=0.46)
    laplacian_img = laplacian(gaussian_img,threshold=0.25)
    imsave('laplacian_'+file,laplacian_img)
    imsave('sobelPrewitt_'+file,sobelPrewitt_img)

    file = 'Coins.png'
    img = imread(file)
    gaussian_img = gaussian(img,5,5)
    sobelPrewitt_img = sobelPrewitt(gaussian_img,threshold=0.2)
    laplacian_img = laplacian(gaussian_img,threshold=0.01)
    imsave('laplacian_'+file,laplacian_img)
    imsave('sobelPrewitt_'+file,sobelPrewitt_img)

    file = 'NoisyCoins.png'
    img = imread(file)
    gaussian_img = gaussian(img,5,5)
    sobelPrewitt_img = sobelPrewitt(gaussian_img,threshold=0.2)
    laplacian_img = laplacian(gaussian_img,threshold=0.006)
    imsave('laplacian_'+file,laplacian_img)
    imsave('sobelPrewitt_'+file,sobelPrewitt_img)

