##### 
#
# Creates a local copy of converted data in numpy's npz format.
#
# Before running training and evaluation:
# Save raw fashion mnist dataset files in the same directory, change into that directory and run the following in command line:
# python fashion.py
#
#####

import numpy as np
from scipy import ndimage

import gzip

fn_xtrain = "train-images-idx3-ubyte.gz"
fn_ytrain = "train-labels-idx1-ubyte.gz"
fn_xtest = "t10k-images-idx3-ubyte.gz"
fn_ytest = "t10k-labels-idx1-ubyte.gz"

destination_xtrain = "fashion/xtrain.npz"
destination_ytrain = "fashion/ytrain.npz"
destination_xtest = "fashion/xtest.npz"
destination_ytest = "fashion/ytest.npz"

file = gzip.open(fn_xtrain,'r')
image_size = 28
num_images = 60000
file.read(16)
buf = file.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_train = data.reshape(num_images, image_size, image_size, 1)

file = gzip.open(fn_ytrain,'r')
num_labels = 60000
file.read(8)
buf = file.read(num_labels)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
Y_train = labels 
label_train = Y_train

file = gzip.open(fn_xtest,'r')
image_size = 28
num_images = 10000
file.read(16)
buf = file.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_test = data.reshape(num_images, image_size, image_size, 1)

file = gzip.open(fn_ytest,'r')
num_labels = 10000
file.read(8)
buf = file.read(num_labels)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
Y_test = labels
label_test = Y_test

images_train = X_train/255.
images_test = X_test/255.

np.savez_compressed(destination_xtrain, images_train)
np.savez_compressed(destination_ytrain, label_train)
np.savez_compressed(destination_xtest, images_test)
np.savez_compressed(destination_ytest, label_test)