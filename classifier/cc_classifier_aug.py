#============================================================================================
# Classifier Training
# Author: Gerald M
#
# This script uses a convolution neural network classifier, to train for cells and non-cell
# objects. This incorporates the Augmentor package to improve augmentation data.
#============================================================================================

import os
import random
import shutil
import datetime
from multiprocessing import cpu_count
import skimage.exposure

import Augmentor

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
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
# Precheck on directory structure
#=============================================================================================

if os.listdir('8-bit/test_data/cell/'):
    print 'Warning! Files detected in test data directory!'
    print 'Moving files back to training data directory...'

    for f in os.listdir('8-bit/test_data/cell/'):
        shutil.move('8-bit/test_data/cell/'+f,'8-bit/training_data/cell/'+f)

    for f in os.listdir('8-bit/test_data/nocell/'):
        shutil.move('8-bit/test_data/nocell/'+f,'8-bit/training_data/nocell/'+f)

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
classifier.add(Conv2D(8, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))

# Pooling operation
# Pooling reduces size of images in order to reduce number of nodes
# 2x2 matrix
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Batch normalization
# Normalize the activations of the previous layer at each batch
# i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
classifier.add(BatchNormalization())

classifier.add(Conv2D(16, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())

classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())

# Prevent nodes from generating same classifiers
classifier.add(Dropout(0.5))

# Flattening operation
# Convert pooled images into vectors
classifier.add(Flatten())

# Connect the set of nodes which input to connected layers
# Units is number of nodes
classifier.add(Dense(units = 128, activation = 'relu'))

# Prevent nodes from generating same classifiers
classifier.add(Dropout(0.5))

# Initialise output layer
classifier.add(Dense(units = 2, activation = 'softmax')) #sigmoid

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

#=============================================================================================
# Augmenting data
#=============================================================================================

training_datagen = Augmentor.Pipeline('8-bit/training_data')

training_datagen.rotate_without_crop(probability=0.5, max_left_rotation=25, max_right_rotation=25, expand=False)
training_datagen.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
training_datagen.flip_left_right(probability=0.5)
training_datagen.flip_top_bottom(probability=0.5)
training_datagen.skew(probability=0.5, magnitude=0.3)
training_datagen.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
training_datagen.shear(probability=0.5,  max_shear_left=2, max_shear_right=2)
training_datagen.random_contrast(probability=0.5, min_factor=0.3, max_factor=1.)

training_data = training_datagen.keras_generator(batch_size=32)

# test data
test_datagen = Augmentor.Pipeline('8-bit/test_data')

test_datagen.rotate_without_crop(probability=0.5, max_left_rotation=25, max_right_rotation=25, expand=False)
test_datagen.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
test_datagen.flip_left_right(probability=0.5)
test_datagen.flip_top_bottom(probability=0.5)
test_datagen.skew(probability=0.5, magnitude=0.3)
test_datagen.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
test_datagen.shear(probability=0.5,  max_shear_left=2, max_shear_right=2)
test_datagen.random_contrast(probability=0.5, min_factor=0.3, max_factor=1.)

test_data = test_datagen.keras_generator(batch_size=32)

print "Done!\n"

#=============================================================================================
# Fitting data to model
#=============================================================================================

print "Fitting data to model..."

# Find number of epoch and validation steps
steps_epoch = len([filename for filename in os.listdir('8-bit/training_data/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir('8-bit/training_data/nocell') if filename.endswith(".tif")])//32
steps_valid = len([filename for filename in os.listdir('8-bit/test_data/cell') if filename.endswith(".tif")]) + len([filename for filename in os.listdir('8-bit/test_data/nocell') if filename.endswith(".tif")])//32

# Checkpoint to only save the best model, metric = val_acc
strdate = datetime.datetime.today().strftime('%Y_%m_%d')

if not os.path.exists('models/'+strdate):
    os.makedirs('models/'+strdate)

filepath = "models/"+strdate+"/cc_model_"+strdate+".h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# steps_per_epoch is number of images in training set
classifier.fit_generator(training_data, steps_per_epoch = steps_epoch, epochs = 10, callbacks=callbacks_list, validation_data = test_data, validation_steps = steps_valid, shuffle = True)

print "Done!\n"

#=============================================================================================
# Recompiling training and test data together
#=============================================================================================

for f in os.listdir('8-bit/test_data/cell/'):
    shutil.move('8-bit/test_data/cell/'+f,'8-bit/training_data/cell/'+f)

for f in os.listdir('8-bit/test_data/nocell/'):
    shutil.move('8-bit/test_data/nocell/'+f,'8-bit/training_data/nocell/'+f)

#=============================================================================================
# Splitting final model into
#=============================================================================================

model = load_model(filepath)
# Serialize model to JSON
model_json = model.to_json()
with open(filepath[:-3]+".json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights(filepath)


print "Done!\n"
