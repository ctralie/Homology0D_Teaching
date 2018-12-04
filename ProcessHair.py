import numpy as np
from scipy import ndimage
from ripser import ripser, plot_dgms, lower_star_img
import matplotlib.pyplot as plt
from scipy import sparse
import time
import PIL
from mpl_toolkits.mplot3d import Axes3D
import sys 
import skimage.transform
import glob

directoryName = '.' # Replace this with whatever directory

areaTarget = 250*250 # Target size in pixels of the mask

for f in glob.glob("%s/*.jpg"%directoryName):
    hair_original = plt.imread(f)
    mask = plt.imread("%s_mask.png"%f[0:-4])
    hair_grey = np.asarray(PIL.Image.fromarray(hair_original).convert('L'))
    hair_grey = hair_grey*mask

    # Smoothing (helps with noise, but could destroy structure at fine scales)
    #hair_grey = ndimage.uniform_filter(hair_grey, size=10)


    # Switching from sublevelset to superlevelset
    hair_grey = -hair_grey


    # Normalize to the range [0, 1] so that full contrast goes from 0 to 1
    hair_grey = hair_grey - np.min(hair_grey)
    hair_grey = hair_grey/np.max(hair_grey)
    
    
    #Normalize by the area of the hair mask and resize
    area = np.sum(mask > 0)
    fac = np.sqrt(areaTarget/float(area))
    print("Resizing by a factor of %.3g"%fac**2)
    newSize = (int(hair_grey.shape[0]*fac), int(hair_grey.shape[1]*fac))
    hair_grey = skimage.transform.resize(hair_grey, newSize, mode='constant')


    # Do lower star filtration after adding a little bit of noise
    # The noise is a hack to help find representatives for the classes
    I = lower_star_img(hair_grey)
    I = I[I[:, 1]-I[:, 0] > 0.001, :] # Filter out low persistence values
    
    
    ## TODO: Make two persistence images: one for sublevelset and one for superlevelset
    ## https://github.com/scikit-tda/persim/blob/master/notebooks/Persistence%20Images.ipynb
    ## https://github.com/scikit-tda/persim/blob/master/notebooks/Classification%20with%20persistence%20images.ipynb
    
    
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.title(hair_original.shape)
    plt.imshow(hair_original)
    plt.axis('off')
    plt.subplot(132)
    plt.title(hair_grey.shape)
    plt.imshow(hair_grey, cmap='afmhot')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(133)
    plot_dgms(I)
    plt.tight_layout()
    plt.show()
