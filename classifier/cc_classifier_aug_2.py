import os
import random
import shutil
import datetime
from multiprocessing import cpu_count
import skimage.exposure
import Augmentor
import tensorflow as tf

import keras
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop

os.environ['MKL_NUM_THREADS'] = str(cpu_count())
os.environ['GOTO_NUM_THREADS'] = str(cpu_count())
os.environ['OMP_NUM_THREADS'] = str(cpu_count())
os.environ['openmp'] = 'True'

config = tf.ConfigProto(device_count={"GPU" : 1, "CPU" : cpu_count()})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

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

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(80, 80, 1)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = RMSprop(lr=1e-4)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

print "Done!"

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

training_data = training_datagen.keras_generator(batch_size=32, scaled=True)

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

test_data = test_datagen.keras_generator(batch_size=32, scaled=True)

print "Done!"

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
model.fit_generator(training_data, steps_per_epoch = steps_epoch, epochs = 25, callbacks=callbacks_list, validation_data = test_data, validation_steps = steps_valid, shuffle = True)

print "Done!"

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

#=============================================================================================
# Writing model summary to file
#=============================================================================================

with open('models/'+strdate+'/'+strdate+'_model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

print "Done!"
