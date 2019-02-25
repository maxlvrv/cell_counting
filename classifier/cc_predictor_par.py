#============================================================================================
# Cell Counting Predictor - Parallel version
# Author: Gerald M
#
# This script uses a convolution neural network classifier, trained on manually identified
# cells, to confirm whether a potential cell/neuron is correctly identified.
#
# This version uses a parallel thread.
#
# Installation:
# 1) Navigate to the folder containing cc_predictor_par.py
#
# Instructions:
# 1) Fill in the user defined parameters from line 94
# 2) Run the script in a Python IDE
#============================================================================================

from __future__ import division
import os, sys, warnings, time
import numpy as np
import pandas as pd
from PIL import Image
from xml.dom import minidom
from multiprocessing import Pool, cpu_count, Array, Manager
from contextlib import closing
from functools import partial
import tqdm
import csv
from natsort import natsorted
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import glob

#=============================================================================================
# Define function for predictions
#=============================================================================================

# Appends cell coordinates to manager list - manager allows access from a thread
def append_cell(coord):
    cell_markers.append(coord)

# Appends no-cell coordinates to manager list - manager allows access from a thread
def append_nocell(coord):
    nocell_markers.append(coord)

# Function to predict/classify object as cell or no-cell
def cellpredict(cell, model_weights_path, model_json_path, marker, image_path, filename, cell_markers, nocell_markers):
    # Import modules - required for each independant thread
    import keras
    from keras.preprocessing import image
    from keras.models import load_model, model_from_json

    # Warning supression and allowing large images to be laoded
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    Image.MAX_IMAGE_PIXELS = 1000000000

    # Load the classifier model
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_path)

    # Load each image then crop for the cell
    # img = Image.open(os.path.join(image_path, filename[marker[cell, 2]])).crop((marker[cell, 1]-40, marker[cell, 0]-40, marker[cell, 1]+40, marker[cell, 0]+40))
    img = Image.open(os.path.join(image_path, filename[marker[cell, 2]-1])).crop((marker[cell, 0]-40, marker[cell, 1]-40, marker[cell, 0]+40, marker[cell, 1]+40))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    # Predict 0 or 1
    prediction = model.predict(np.asarray(img))

    if prediction[0][0] == 1: # Cell
        cell_value = 1
        append_cell(marker[cell,:])
        #image.array_to_img(cell_crop[0,:,:,:]).save('/Users/gm515/Desktop/cell_par/'+str(cell)+'.tif')
    else: # No cell
        cell_value = 0
        append_nocell(marker[cell,:])
        #image.array_to_img(cell_crop[0,:,:,:]).save('/Users/gm515/Desktop/nocell_par/'+str(cell)+'.tif')

    result[cell] = cell_value

    return


# Main function
if __name__ == '__main__':
    #=============================================================================================
    # User definied parameters
    #=============================================================================================

    # CNN model paths
    model_weights_path = 'models/2019_01_29/cc_model_2019_01_29.h5'
    model_json_path = 'models/2019_01_29/cc_model_2019_01_29.json'

    # Directory path to the files containing the cell coordinates
    count_path = '/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections/counts'

    # Directory path to the TIFF files containing the cells
    image_path = '/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections'

    #=============================================================================================
    # Loop through the coordinate files and predict cells
    #=============================================================================================

    # Get list of files containing the coordinates in x, y, z
    #image_path = raw_input('Counting file path (drag-and-drop): ').strip('\'').rstrip()
    filename = natsorted([file for file in os.listdir(image_path) if file.endswith('.tif')])

    if os.path.isdir(count_path):
        all_marker_path = glob.glob(count_path+'/*.csv')
    else:
        all_marker_path = count_path

    # Create empty pands dataframe to store data
    df = pd.DataFrame(columns = ['ROI', 'Original', 'True', 'False'])

    for marker_path in all_marker_path:

        marker_filename, marker_file_extension = os.path.splitext(marker_path)

        if marker_file_extension == '.xml':
            xml_doc = minidom.parse(marker_path)

            marker_x = xml_doc.getElementsByTagName('MarkerX')
            marker_y = xml_doc.getElementsByTagName('MarkerY')
            marker_z = xml_doc.getElementsByTagName('MarkerZ')

            marker = np.empty((0,3), int)

            for elem in range (0, marker_x.length):
                marker = np.vstack((marker, [int(marker_x[elem].firstChild.data), int(marker_y[elem].firstChild.data), int(marker_z[elem].firstChild.data)]))
        if marker_file_extension == '.csv':
            marker = np.genfromtxt(marker_path, delimiter=',', dtype=np.float).astype(int)

        #=============================================================================================
        # Load images and correct cell count by predicting
        #=============================================================================================

        manager = Manager()
        result = Array('i', marker.shape[0])
        cell_markers = manager.list()
        nocell_markers = manager.list()

        cell_index = range(marker.shape[0])

        print 'Classifiying in: '+marker_filename

        tstart = time.time()
        pool = Pool(cpu_count())
        res = list(tqdm.tqdm(pool.imap(partial(cellpredict, model_weights_path=model_weights_path, model_json_path=model_json_path, marker=marker, image_path=image_path, filename=filename, cell_markers=cell_markers, nocell_markers=nocell_markers), cell_index), total=marker.shape[0]))

        pool.close()
        pool.join()

        # Append to Pandas dataframe
        df = df.append({'ROI':marker_filename.split('/')[-1][:-9], 'Original': result[:].count(1)+result[:].count(0), 'True': result[:].count(1), 'False': result[:].count(0)}, ignore_index=True)

        # Write dataframe to csv
        df.to_csv(count_path[:-7]+'/counts_cc_corrected.csv', index=False)

print df
print '{0:.0f}:{1:.0f} (MM:SS)'.format(*divmod(time.time()-tstart,60))
