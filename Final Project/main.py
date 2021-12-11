'''
Low Light Image Enhancement of color and grayscale images based on Retinex theory

Adarsh Shah, 19473
Anmol Asati, 

1. Naive Approach
2. Single Scale Retinex 
3. Multi Scale Retinex
4. Multi Scale Retinex with Color Retention
5. MSRCR with Histogram Clipping
6. Bilateral and Gaussian Laplacian Pyramid based Image Enhancement

All the functions are test functions. Some minor modifications can be made to use them in other modules.
'''

import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.io import imread,imsave
from skimage.restoration import denoise_bilateral
from skimage.filters import gaussian
from scipy.signal import convolve2d
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.metrics import peak_signal_noise_ratio,mean_squared_error,structural_similarity
import matplotlib.pyplot as plt

def naiveApproach(file,saveas,true_img):
    '''
    A Naive Approach based on understading to enhance color images

    RGB -> HSV                             HSV' -> RGB' 
             V -> CLAHE(V) -> Bilateral -> V'
    
    Reasoning:
    V contains brightness information (Illumination)
    Histogram of V shows image has low dynamic range
    Improve it by performing CLAHE on V
    Overcome Dark spots using Bilateral filter on V

    Tuning Parameters:
    CLAHE Clip Limit
    CLAHE Window Size
    Bilateral spatial sigma
    Bilateral color sigma

    Result:
    Observed Low Light Enhancement.
    Not all local regions are brightened.
    Color Constancy is maintained.
    '''

    img = img_as_float(imread(file))
    hsv_img = rgb2hsv(img)

    hsv_img[:,:,2] = equalize_adapthist(image=hsv_img[:,:,2],nbins=256,clip_limit=0.01)
    hsv_img[:,:,2] = denoise_bilateral(hsv_img[:,:,2],1,2)

    hsv_img[:,:,2] = rescale_intensity(hsv_img[:,:,2],out_range=(0,1.0))

    img2 = hsv2rgb(hsv_img)
    true_img = img_as_float(imread(true_img))
    psnr = peak_signal_noise_ratio(true_img,img2)
    mss = structural_similarity(true_img,img2,multichannel=True)
    mse = mean_squared_error(true_img,img2)
    print(f'Naive Approach : MSE:{mse:.2f} || PSNR:{psnr:.2f} || MSS:{mss:.2f}')

    imsave(f'{saveas}.png',img_as_ubyte(hsv2rgb(hsv_img)))


def SSR(file,sigma,saveas,true_img):
    '''
    file : input file
    sigma : GSF standard deviation
    saveas : filename to saveas
    Single Scale Retinex

    Reasoning:
    Seperate Illumination from Intensity using a Gaussian Kernel
    Intensity Information without Illumination is ideally stored in Reflectance
    Reflectance = log(Intensity) - log(Intensity*Gaussian)
    Save Reflectance

    Based on https://ieeexplore.ieee.org/document/6176791

    Result:
    Observed low light enhancement.
    Darker Patches from the naive approach are lightened.
    Color Constancy is not maintained. Graying occurs.
    As the standard deviation of GSF is increased, graying decreases, color constancy increases but can cause halo effects.
    '''

    img = img_as_float(imread(file))
   

    reflectance = np.zeros(np.shape(img))

    s = sigma

    illumination = gaussian(image=img,sigma=s,multichannel=True)
    reflectance = np.log(1+img)-np.log(1+illumination)
    
    reflectance = rescale_intensity(reflectance,out_range=(0,255)).astype(np.ubyte)

    true_img = imread(true_img)
    psnr = peak_signal_noise_ratio(true_img,reflectance)
    mss = structural_similarity(true_img,reflectance,multichannel=True)
    mse = mean_squared_error(true_img,reflectance)
    print(f'SSR : MSE:{mse:.2f} || PSNR:{psnr:.2f} || MSS:{mss:.2f}')

    imsave(fname=f'{saveas}.png',arr=reflectance)

def MSR(file,sigma,saveas,true_img):
    '''
    file : input file
    sigma : ndarray of sigmas
    saveas : filename to saveas
    Multi Scale Retinex

    Reasoning:
    Perform Single Scale for multiple sigmas.
    Take weighted average of the result.

    Based on https://ieeexplore.ieee.org/document/6176791

    Result:
    Observed low light enhancement
    Color Constancy is not maintained. Graying Occurs.
    '''

    img = img_as_float(imread(file))

    reflectance = np.zeros(np.shape(img))

    for s in sigma:
        illumination = gaussian(image=img,sigma=s,multichannel=True)
        reflectance += (np.log(1+img)-np.log(1+illumination))/3

    reflectance = rescale_intensity(reflectance,out_range=(0,255)).astype(np.ubyte)

    true_img = imread(true_img)
    psnr = peak_signal_noise_ratio(true_img,reflectance)
    mss = structural_similarity(true_img,reflectance,multichannel=True)
    mse = mean_squared_error(true_img,reflectance)
    print(f'MSR : MSE:{mse:.2f} || PSNR:{psnr:.2f} || MSS:{mss:.2f}')

    imsave(fname=f'{saveas}.png',arr=reflectance)


def MSRCR(file,saveas,true_img):
    '''
    Multi Scale Retinex with Color Retention

    Reasoning:

    Based on https://ieeexplore.ieee.org/document/6176791

    Result:
    No significant difference
    Graying still persists
    Color correction is required.
    '''

    '''
    The values based on
    D. J. Jobson, Zia-ur-Rahman, G. A. Woodell, "Properties and performance of a Center/Surround Retinex," IEEE Transactions on Image Processing, vol. 6, no. 3, March 1997.
    '''
    G = 192
    alpha = 125
    beta = 46
    b = -30

    img = img_as_float(imread(file))

    reflectance = np.zeros(np.shape(img))

    for s in [70,150,200]:
        illumination = gaussian(image=img,sigma=s,multichannel=True)
        reflectance += (np.log(1+img)-np.log(1+illumination))/3
    
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_info = beta * (np.log10(alpha * img + 1) - np.log10(img_sum + 1))

    reflectance = G * (color_info*reflectance+b)

    reflectance = rescale_intensity(reflectance,out_range=(0,255)).astype(np.ubyte)

    true_img = imread(true_img)
    psnr = peak_signal_noise_ratio(true_img,reflectance)
    mss = structural_similarity(true_img,reflectance,multichannel=True)
    mse = mean_squared_error(true_img,reflectance)
    print(f'MSRCR : MSE:{mse:.2f} || PSNR:{psnr:.2f} || MSS:{mss:.2f}')

    imsave(fname=f'{saveas}.png',arr=reflectance)

def MSRCR_hist_clipping(file,saveas,true_img):
    '''
    This function is basically doing Histogram clipping after performing Retinex algorithm on image

    Input- Retinex algo Output
    Output- Histogram equalized Image

    Based on https://ieeexplore.ieee.org/document/6176791

    Observations:
    significant difference
    Graying reduced
    some cliiping parameter tuning required (LCP/UCP)
    ''' 

    G = 192
    alpha = 125
    beta = 46
    b = -30

    img = img_as_float(imread(file))

    reflectance = np.zeros(np.shape(img))

    for s in [70,150,200]:
        illumination = gaussian(image=img,sigma=s,multichannel=True)
        reflectance += (np.log(1+img)-np.log(1+illumination))/3
    
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_info = beta * (np.log10(alpha * img + 1) - np.log10(img_sum + 1))

    reflectance = G * (color_info*reflectance+b)
    
    img = rescale_intensity(reflectance,out_range=(0,255)).astype(np.ubyte)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    #center = hist[125] # improvement
    center = np.mean(hist)
    hist = np.where(hist <= center*0.05 , 0 , hist)

    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    reflectance = cdf[img]

    true_img = imread(true_img)
    psnr = peak_signal_noise_ratio(true_img,reflectance)
    mss = structural_similarity(true_img,reflectance,multichannel=True)
    mse = mean_squared_error(true_img,reflectance)
    print(f'MSR : MSE:{mse:.2f} || PSNR:{psnr:.2f} || MSS:{mss:.2f}')

    imsave(fname=f'{saveas}.png',arr=reflectance)
    


def GaussianLaplacianPyramid(image,sigma_spatial,sigma_color,sigma_LG,N):
    '''
    Used in Bilateral based Retinex Algorithm

    Based on : https://doi.org/10.3390/a12120258

    Generate Gaussian Scale Space of Images
    Apply Bilateral filter to each Gaussian Image
    Perform Laplace Reconstruction
    '''
    scaleSpace = list()

    img = img_as_float(image)
    scaleSpace.append(img)

    for _ in range(N):
        P,Q,x = scaleSpace[-1].shape
        P,Q = int(P/2),int(Q/2)
        img = resize(gaussian(scaleSpace[-1],sigma_LG,multichannel=True),(P,Q),1)
        img = denoise_bilateral(img,sigma_color=sigma_color,sigma_spatial=sigma_spatial,multichannel=True)
        scaleSpace.append(img)


    l = len(scaleSpace)-1
    while l>=1:
        Fl = scaleSpace[l-1]
        Fl_1 = scaleSpace[l]
        Dl = resize(Fl_1,Fl.shape,order=1)
        Fn = Fl-Dl
        Fn = Fl+Fn
        scaleSpace[l-1] = Fn
        l-=1
    
    return scaleSpace[0]


def BFR(file,sigma_spatial,sigma_color,sigma_LG,N,saveas,true_img):
    '''
    Low Light Image enhancement based on Retinex and Bilateral Filtering

    Based on : https://doi.org/10.3390/a12120258
    Improved Bilateral Filtering for a Gaussian Pyramid Structure-Based Image Enhancement Algorithm

    Result : 
    Better Edge Retention with reduced dark regions in the enhanced image.
    '''

    img = img_as_float(imread(file))
    hsv_img = rgb2hsv(img)

    hsv_img[:,:,2] = denoise_bilateral(image = hsv_img[:,:,2],sigma_color=sigma_color,sigma_spatial=sigma_spatial)
    hsv_img[:,:,2] = rescale_intensity(hsv_img[:,:,2],out_range=(0,1))

    img = hsv2rgb(hsv_img)
    img = GaussianLaplacianPyramid(img,sigma_spatial,sigma_color,sigma_LG,N)

    img = rescale_intensity(img,out_range=(0,255)).astype(np.ubyte)
    
    img_true = imread(true_img)
    psnr = peak_signal_noise_ratio(img_true,img)
    mss = structural_similarity(img_true,img,multichannel=True)
    mse = mean_squared_error(img_true,img)
    print(f'BFR : MSE:{mse:.2f} || PSNR:{psnr:.2f} || MSS:{mss:.2f}')

    imsave(f'{saveas}.png',img)

if __name__=='__main__':

    '''
    Analysis of Single Scale Retinex:

    The Gaussian Function's variance is the tuning parameter. 
    It is varied over the range 100->1000 and the results are saved in SSR folder.
    Key Observation: Increasing variance reduces graying in the enhanced image but some local regions get darker.
    
    '''
    for sigma in np.arange(50,501,50):
        SSR('low_women.png',sigma,f'SSR/SSR_{sigma}','actual_women.png')

    '''
    Analysis of Multi Scale Retinex:

    Key Observation: Improved color constancy and color retention as both lower and higher variance based SSR outputs are combined.
    '''
    for sigma_low in np.arange(10,101,10):
        for sigma_high in np.arange(50,500,50):
            if sigma_low<sigma_high:
                MSR('low_women.png',np.linspace(int(sigma_low),int(sigma_high),3),f'MSR/MSR_{int(sigma_low)}_{int(sigma_high)}','actual_women.png')

    '''
    Analysis of Bilateral based Image Enhancement Algorithm
    Key Observation: edge retention, denoising, enhancement of low light regions with color retention to some extent. Graying obsereved in some area.
    '''
    for s1 in [5,7,10,15]:
         for r in [3,5,7,9,15,20]:
             for s2 in [50,100,250,500,750,1000]:
                 if r<s1:
                    BFR('low_women.png',s1,r/255,s2,5,f'BFR/BFR_{s1}_{r}_{s2}','actual_women.png')


    SSR('low_women.png',400,f'SSR/SSR_{400}','actual_women.png')
    MSR('low_women.png',np.linspace(20,250,3),f'MSR/MSR_{20}_{250}','actual_women.png')
    MSRCR('low_women.png','MSRCR/MSRCR','actual_women.png')
    MSRCR_hist_clipping('low_women.png','MSRCRwithHistogramClipping/MSRCRwithHistogramClipping','actual_women.png')
    BFR('low_women.png',15,3/255,1000,5,f'BFR/BFR_{s1}_{r}_{s2}','actual_women.png')








