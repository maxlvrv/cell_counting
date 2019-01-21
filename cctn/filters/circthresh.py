#============================================================================================
# Circularity Based Thresholding Function
# Author: Gerald M
#
# This function returns the threshold at which the image has the highest mean circularity.
#============================================================================================

import scipy.ndimage
from scipy.optimize import curve_fit
import numpy as np
import math, csv
from multiprocessing import Pool, cpu_count, Array, Manager
from functools import partial
from skimage.measure import regionprops, label
from PIL import Image
import matplotlib.pyplot as plt

def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def circularity(thresh, A, SIZE):
    A_thresh = (A>thresh).astype(int)
    A_thresh = scipy.ndimage.morphology.binary_fill_holes(A_thresh).astype(int)

    #Image.fromarray(A_thresh.astype(float)).save('/Users/gm515/Desktop/temp/circ/'+str(thresh)+'.tif')

    A_label = label(A_thresh, connectivity=A_thresh.ndim)

    # Find circularity
    circfunc = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

    circ = [circfunc(region) for region in regionprops(A_label) if region.area>SIZE and region.area<SIZE*10 and region.perimeter>0]

    if len(circ)>0:
        return np.mean(np.array(circ))
    else:
        return 0.

def circthresh(A,SIZE,THRESHLIM):
    thresh_int = 2
    thresh_all = np.arange(np.min(A),np.max(A),thresh_int)

    # Get mean circularity
    pool = Pool(cpu_count())
    circ_all = np.squeeze(np.array([pool.map(partial(circularity, A=A, SIZE=SIZE), thresh_all)]), axis=0)
    pool.close()
    pool.join()

    # Fit a polynomial and find optimum threshold where circualrity measures are grater than 1
    # func = np.polyfit(thresh_all[circ_all>0], circ_all[circ_all>0], 2)
    # yfit = np.poly1d(func)
    # print yfit(thresh_all[circ_all>0])

    plot = False
    if plot:
        plt.figure()
        plt.plot(thresh_all, circ_all, 'xb')
        plt.plot(thresh_all[circ_all>0], yfit(thresh_all[circ_all>0]), '-r')
        plt.savefig('/Users/gm515/Desktop/test/fig.png')

    #T = thresh_all[np.where(yfit(thresh_all[circ_all>0]) == np.max(yfit(thresh_all[circ_all>0])))]
    T = thresh_all[np.argmax(circ_all>0.8)]

    # If threshold from fit is less than minimum threshold limit, set optimum threshold to minimum
    if T<THRESHLIM:
        T = THRESHLIM

    return T
