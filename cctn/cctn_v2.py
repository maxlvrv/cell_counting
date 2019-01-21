#============================================================================================
# Cell Counting in Target Nuclei Script
# Author: Gerald M
#
# This script performs automated cell counting in anatomical structures of interest, or a
# a stack of TIFFs. It works by first determining an ideal threshold based on the circularity
# of objects. Then by tracking cells/objects over multiple layers to account for oversampling.
# The output provides a list of coordinates for identified cells. This should then be fed
# into the image predictor to confirm whether objects are cells or not.
#
# Version 2 - v2
# This version differes from original by removing all empty rows and columns to further
# crop each image. In addition, a rolling ball background subtration is used to remove
# uneven background and generally help the cell segmentation process.
#
# Instructions:
# 1) Go to the user defined parameters from roughly line 80
# 2) Make changes to those parameters as neccessary
# 3) Execute the code in a Python IDE
#============================================================================================

################################################################################
## Module import
################################################################################

import os, time, numpy, math, json, warnings, csv, sys, collections, cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.misc as sc
import scipy.ndimage as ndimage
from filters.gaussmedfilt import gaussmedfilt
from filters.medfilt import medfilt
from filters.circthresh import circthresh
from skimage.measure import regionprops, label
from PIL import Image
from skimage import io
from natsort import natsorted
from filters.rollingballfilt import rolling_ball_filter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

################################################################################
## Function definitions
################################################################################

def distance(a, b):
    return (a[0] - b[0])**2  + (a[1] - b[1])**2

def get_children(json_obj, acr, ids):
    for obj in json_obj:
        if obj['children'] == []:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
        else:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
            get_children(obj['children'], acr, ids)
    return (acr, ids)

def get_structure(json_obj, acronym):
    found = (False, None)
    for obj in json_obj:
        if obj['acronym'].lower() == acronym:
            #print obj['acronym'], obj['id']
            [acr, ids] = get_children(obj['children'], [], [])
            #print ids
            if ids == []:
                acr = [obj['acronym']]
                ids = [obj['id']]
                return (True, acr, ids)
            else:
                acr.append(obj['acronym'])
                ids.append(obj['id'])
                return (True, acr, ids)
        else:
            found = get_structure(obj['children'], acronym)
            if found:
                return found

if __name__ == '__main__':
    ################################################################################
    ## User defined parameters - please fill in the parameters in this section only
    ################################################################################

    # Do you want to use a mask taken from a registered segmentation atlas
    mask = True

    # Do you want to perform over sampling correction?
    # Cells within a radius on successive images will be counted as one cell
    over_sample = True

    # The following is redundant and will be included when considering volume and density
    # xy_res = 10
    # z_res = 5

    # If you are using a mask, input the mask path and the structures you want to count within
    # E.g. 'LGd, LGv, IGL, RT'
    if mask:
        mask_path = '/mnt/TissueCyte80TB/181012_Gerald_KO/ko-Mosaic/SEGMENTATION_RES.tif'
        structure_list = 'LGd'#,LGv,IGL,RT,LP,VPM,VPL,APN,ZI,LD'

    # Input details for the cell morphology
    # Can be left as default values
    size = 85.
    radius = 12.

    # Input the directory path of the TIFF images for counting
    count_path = '/mnt/TissueCyte80TB/181012_Gerald_KO/ko-Mosaic/Ch2_Stitched_Sections'

    # Of the images in the above directory, how many will be counted?
    # Number of files [None,None] for all, or [start,end] for specific range
    number_files = [None,None]

    # Do you want to use the custom donut median filter?
    use_medfilt = False

    # For the circularity threshold, what minimum background threshold should be set
    # You can estimate this by loading an image in ImageJ, perform a gaussian filter radius 3, then perform a rolling ball background subtraction radius 8, and choose a threshold which limits remaining background signal
    bg_thresh = 5

    ################################################################################
    ## Initialisation
    ################################################################################

    # Create directory to hold the counts in same folder as the images
    if not os.path.exists(count_path+'/counts'):
        os.makedirs(count_path+'/counts')

    # List of files to count
    count_files = []
    count_files += [each for each in os.listdir(count_path) if each.endswith('.tif')]
    count_files = natsorted(count_files)
    if number_files[0] != None:
        count_files = count_files[number_files[0]-1:number_files[1]]
    print 'Counting in files: '+count_files[0]+' to '+count_files[-1]

    ################################################################################
    ## Retrieving structures IDs
    ################################################################################

    if mask:
        #path = raw_input('NII/TIFF file path (drag-and-drop): ').rstrip()
        file, extension = os.path.splitext(mask_path)
        if extension == '.nii':
            seg = nib.load(mask_path).get_data()
        else:
            seg = io.imread(mask_path)
        print 'Loaded segmentation data'

    ids = []
    acr = []
    if mask:
        anno_file = json.load(open('2017_annotation_structure_info.json'))
        structure_list = [x.strip() for x in structure_list.lower().split(",")]
        for elem in structure_list:
            a, i = get_structure(anno_file['children'], elem)[1:]
            ids.extend(i)
            acr.extend(a)
    else:
        ids.extend('none')
        acr.extend('none')
    print 'Counting in structures: '+str(acr)

    ################################################################################
    ## Counting
    ################################################################################

    tstart = time.time()

    temp = Image.open(count_path+'/'+count_files[0])
    temp_size = temp.size
    temp = None
    if mask:
        scale = float(temp_size[1])/seg[1].shape[0]

    structure_index = 0

    for name, structure in zip(acr,ids):
        print 'Counting in '+str(name)
        proceed = True

        # Dictionary to store centroids - each key is a new slice number
        total_cells = dict()

        ################################################################################
        ## Obtain crop information for structure if mask required
        ################################################################################
        if mask:
            index = np.array([[],[],[]])
            if structure in seg:
                index = np.concatenate((index, np.array(np.nonzero(structure == seg))), axis=1)
            else:
                proceed = False

            if index.size > 0:
                zmin = int(index[0].min())
                zmax = int(index[0].max())
            else:
                proceed = False
        else:
            zmin = 0
            zmax = len(count_files)

        ################################################################################
        ## Check whether to proceed with the count
        ################################################################################
        if proceed:
            ################################################################################
            ## Loop through slices based on cropped boundaries
            ################################################################################
            for slice_number in range(zmin,zmax):
                # Load image and convert to dtype=float and scale to full 255 range
                image = Image.open(count_path+'/'+count_files[slice_number])
                image = np.array(image).astype(float)
                image = np.multiply(np.divide(image,np.max(image)), 255.)

                # Apply mask if required
                if mask:
                    mask_image = np.array(Image.fromarray(seg[slice_number]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))
                    mask_image[mask_image!=structure] = 0
                    image[mask_image==0] = 0

                    # Crop image with idx
                    mask_image = image>0
                    idx = np.ix_(mask_image.any(1),mask_image.any(0))
                    mask_image = None
                    row_idx = idx[0].flatten()
                    col_idx = idx[1].flatten()
                    image = image[idx]
                    #Image.fromarray(np.uint8(image)*255).save('/home/gm515/Documents/Temp4/Z_'+str(slice_number)+'.tif')

                # Perform gaussian donut median filter
                if use_medfilt:
                    image = gaussmedfilt(image, 3, 1.5)
                else:
                    image = ndimage.gaussian_filter(image, sigma=(3, 3))

                if image.shape[0] != 0 and np.max(image) != 0.:
                    #image = np.multiply(np.divide(image, np.max(image)), 255.)
                    # Perform rolling ball background subtraction to remove uneven background signal
                    image, background = rolling_ball_filter(np.uint8(image), 8)
                    #Image.fromarray(image).save('/home/gm515/Documents/Temp3/Z_'+str(slice_number+1)+'.tif')

                    # Perform circularity threshold
                    image = image>circthresh(image,size,bg_thresh)
                    #Image.fromarray(np.uint8(image)*255).save('/home/gm515/Documents/Temp3/Z_'+str(slice_number+1)+'.tif')

                    # Remove objects smaller than chosen size
                    image_label = label(image, connectivity=image.ndim)

                    circfunc = lambda r: (4 * math.pi * r.area) / ((r.perimeter * r.perimeter) + 0.00000001)

                    # Centroids returns (row, col) so switch over
                    circ = [circfunc(region) for region in regionprops(image_label)]
                    areas = [region.area for region in regionprops(image_label)]
                    labels = [region.label for region in regionprops(image_label)]
                    centroids = [region.centroid for region in regionprops(image_label)]

                    # Convert coordinate of centroid to coordinate of whole image
                    coordfunc = lambda celly, cellx : (row_idx[celly], col_idx[cellx])

                    # (row, col) or (y, x)
                    centroids = [coordfunc(int(c[0]), int(c[1])) for c in centroids]

                    #image = np.full(image.shape, False)

                    # Threshold the objects based on size and circularity and store centroids
                    cells = []
                    for i, _ in enumerate(areas):
                        if areas[i] > size/2.5 and areas[i] < size*10 and circ[i] > 0.65:
                            # (row, col) centroid
                            # So flip the order for (col, row) as (x, y)
                            cells.append(centroids[i][::-1])
                            #image += image_label==labels[i]
                else:
                    cells = []

                total_cells.update({slice_number : cells})

                sys.stdout.write("\r%d%%" % int(100*np.float(slice_number-zmin)/(zmax-zmin)))
                sys.stdout.flush()

                #image = image>0 # Create image if needed
                #Image.fromarray(np.uint8(image)*255).save('/home/gm515/Documents/Temp4/Z_'+str(slice_number)+'.tif')

            # Sort the dictionary so key are in order
            total_cells = collections.OrderedDict(sorted(total_cells.items()))

            # Peform correction for overlap
            if over_sample:
                print 'Correcting for oversampling'
                for index, (x, y) in enumerate(zip(total_cells.values()[1:], total_cells.values()[:-1])):
                    if len(x) * len(y) > 0:
                        total_cells[zmin+index] = [ycell for ycell in y if all(distance(xcell,ycell)>radius**2 for xcell in x)]

            num_cells = sum(map(len, total_cells.values()))

            print str(name)+' '+str(num_cells)

            csv_file = count_path+'/counts/'+str(name)+'_v2_count.csv'
            with open(csv_file, 'w+') as f:
                for key in sorted(total_cells.keys()):
                    if len(total_cells[key])>0:
                        csv.writer(f, delimiter=',').writerows(np.round(np.concatenate(([( (np.array(val))).tolist() for val in total_cells[key]], np.ones((len(total_cells[key]), 1))*(key+1)), axis=1)))

        structure_index += 1

    print '~Fin~'

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    print 'Counting completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)
