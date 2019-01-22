#============================================================================================
# Cell Counting Predictor - Parallel version
# Author: Gerald M
#
# This script uses a convolution neural network classifier, trained on manually identified
# cells, to confirm whether a potential cell/neuron is correctly identified.
#
#
# Installation:
# 1) Navigate to the folder containing cc_predictor_par_thread.py
#
# Instructions:
# 1) Run the script in a Python IDE
# 2) Fill in the parameters that you are asked for
#    Note: You can drag and drop folder paths (works on MacOS) or copy and paste the paths
#    Note: The temporary directory is required to speed up ImageJ loading of the files
#============================================================================================

from __future__ import division
import os, sys, warnings, time
import numpy as np
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
    print 'Loading model'
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_path)
    print 'Done'


    # Load each image then crop for the cell
    img = Image.open(os.path.join(image_path, filename[marker[cell, 2]])).crop((marker[cell, 1]-40, marker[cell, 0]-40, marker[cell, 1]+40, marker[cell, 0]+40))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    # Predict 0 or 1
    prediction = model.predict(np.asarray(img))

    if prediction[0][0] == 0: # Cell
        cell_value = 1
        append_cell(marker[cell,:])
        #image.array_to_img(cell_crop[0,:,:,:]).save('/Users/gm515/Desktop/cell_par/'+str(cell)+'.tif')
    else: # No cell
        cell_value = 0
        append_nocell(marker[cell,:])
        #image.array_to_img(cell_crop[0,:,:,:]).save('/Users/gm515/Desktop/nocell_par/'+str(cell)+'.tif')

    result[cell] = cell_value
    print 'Cell classified'
    return


# Main function
if __name__ == '__main__':
    #=============================================================================================
    # User definied parameters
    #=============================================================================================

    # CNN model paths
    model_weights_path = 'old_models/cc_weights_2018_11_20.h5'
    model_json_path = 'old_models/cc_json_2018_11_20.json'

    # Directory path to the files containing the cell coordinates
    count_path = '../TestSection/counts'

    # Directory path to the TIFF files containing the cells
    image_path = '../TestSection'

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
        res = pool.map(partial(cellpredict, model_weights_path=model_weights_path, model_json_path=model_json_path, marker=marker, image_path=image_path, filename=filename, cell_markers=cell_markers, nocell_markers=nocell_markers), cell_index)

        # for i, _ in enumerate(pool.map(partial(cellpredict, model_weights_path=model_weights_path, model_json_path=model_json_path, marker=marker, image_path=image_path, filename=filename, cell_markers=cell_markers, nocell_markers=nocell_markers), cell_index), 1):
        #     sys.stderr.write('\rDone {0:%}'.format(i/len(cell_index)))

        pool.close()
        pool.join()

        print 'Correct cell preditions:', result[:].count(1)
        print 'Potential false cell predictions:', result[:].count(0)


print '{0:.0f}:{1:.0f} (MM:SS)'.format(*divmod(time.time()-tstart,60))
