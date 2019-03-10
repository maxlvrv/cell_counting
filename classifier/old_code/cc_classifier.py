#============================================================================================
# Classifier Training
# Author: Gerald M
#
# This script uses a convolution neural network classifier, to train for cells and non-cell
# objects.
#============================================================================================

import os
import random
import shutil
import datetime
from multiprocessing import cpu_count
import skimage.exposure

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import tensorflow as tf

os.environ['MKL_NUM_THREADS'] = str(cpu_count())
os.environ['GOTO_NUM_THREADS'] = str(cpu_count())
os.environ['OMP_NUM_THREADS'] = str(cpu_count())
os.environ['openmp'] = 'True'

config = tf.ConfigProto(device_count={"GPU" : 1, "CPU" : cpu_count()})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def AHE(img):
    img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq

#=============================================================================================
# Construction of Convolution Neural Network
#=============================================================================================

print "Constructing Convolution Neural Network..."

# Create object of sequential class
classifier = Sequential()

# Convolution operation
# Number of filters
# Shape of each filter
# Input shape in first two, input type in third, 3 = RGB
# activation function, relu = rectifier function
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))

# Pooling operation
# Pooling reduces size of images in order to reduce number of nodes
# 2x2 matrix
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening operation
# Convert pooled images into vectors
classifier.add(Flatten())

# Connect the set of nodes which input to connected layers
# Units is number of nodes
classifier.add(Dense(units = 128, activation = 'relu'))

# Prevent nodes from generating same classifiers
classifier.add(Dropout(0.5))

# Initialise output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile
# Optimizer is stochastic gradient descent
# Loss is loss function
# Metric is performance metric
optimizer = RMSprop(lr=1e-4)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

print "Done!\n"

#=============================================================================================
# Preparing training and test data
#=============================================================================================

from keras.preprocessing.image import ImageDataGenerator

print "Splitting all training data into 70% training and 30% test data directories..."

cell_data = os.listdir('8-bit/training_data/cell/')
nocell_data = os.listdir('8-bit/training_data/nocell/')

test_cell_data = random.sample(cell_data, int(0.3*len(cell_data)))
test_nocell_data = random.sample(nocell_data, int(0.3*len(nocell_data)))

for f in test_cell_data:
    shutil.move('8-bit/training_data/cell/'+f,'8-bit/test_data/cell/'+f)

for f in test_nocell_data:
    shutil.move('8-bit/training_data/nocell/'+f,'8-bit/test_data/nocell/'+f)

# training data
train_datagen = ImageDataGenerator(rotation_range=180, rescale = 1./255, shear_range = 0.15, zoom_range = 0.15, width_shift_range=0.15, height_shift_range=0.15, horizontal_flip = True, vertical_flip = True)
training_data = train_datagen.flow_from_directory('8-bit/training_data', target_size = (80, 80), batch_size = 32, class_mode = 'binary', color_mode = 'grayscale')
#training_data = train_datagen.flow_from_directory('8-bit/training_data', target_size = (80, 80), batch_size = 32, class_mode = 'binary', color_mode = 'grayscale', save_to_dir='preview', save_prefix='cell', save_format='jpeg')
# test data
test_datagen = ImageDataGenerator(rescale = 1./255)
test_data = test_datagen.flow_from_directory('8-bit/test_data', target_size = (80, 80), batch_size = 32, class_mode = 'binary', color_mode = 'grayscale')

print "Done!\n"

#=============================================================================================
# Fitting data to model
#=============================================================================================

print "Fitting data to model..."

# Find number of epoch and validation steps
steps_epoch = len([filename for filename in os.listdir('8-bit/training_data/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir('8-bit/training_data/nocell') if filename.endswith(".tif")])
steps_valid = len([filename for filename in os.listdir('8-bit/test_data/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir('8-bit/test_data/nocell') if filename.endswith(".tif")])

# Checkpoint to only save the best model, metric = val_acc
filepath = "cc_model_"+datetime.datetime.today().strftime('%Y_%m_%d')+".h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# steps_per_epoch is number of images in training set
classifier.fit_generator(training_data, steps_per_epoch = steps_epoch, epochs = 50, callbacks=callbacks_list, validation_data = test_data, validation_steps = steps_valid, shuffle = True)

print "Done!\n"

#=============================================================================================
# Recompiling training and test data together
#=============================================================================================

for f in os.listdir('8-bit/test_data/cell/'):
    shutil.move('8-bit/test_data/cell/'+f,'8-bit/training_data/cell/'+f)

for f in os.listdir('8-bit/test_data/nocell/'):
    shutil.move('8-bit/test_data/nocell/'+f,'8-bit/training_data/nocell/'+f)

print "Done!\n"
