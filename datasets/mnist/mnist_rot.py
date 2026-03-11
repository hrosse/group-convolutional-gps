##### 
#
# Creates a randomly rotated version of the mnist dataset and locally saves data in numpy's npz format.
#
# Before running training and evaluation:
# Save raw mnist dataset files in the same directory, change into that directory and run the following in command line:
# python mnist_rot.py
#
#####

import numpy as np
from scipy import ndimage

import gzip

fn_xtrain = "train-images-idx3-ubyte.gz"
fn_ytrain = "train-labels-idx1-ubyte.gz"
fn_xtest = "t10k-images-idx3-ubyte.gz"
fn_ytest = "t10k-labels-idx1-ubyte.gz"

destination_xtrain = "mnist_rot/xtrain.npz"
destination_ytrain = "mnist_rot/ytrain.npz"
destination_xtest = "mnist_rot/xtest.npz"
destination_ytest = "mnist_rot/ytest.npz"

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
mnist_label_train = Y_train

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
mnist_label_test = Y_test

np.random.seed(101)

xtrain_list = list()

for img in X_train:
    xtrain_list.append(ndimage.rotate(
        img,
        angle=np.random.uniform(0, 360),
        reshape=False,
        mode="constant",
        cval=0.,
        prefilter=False
    ))
    
xtest_list = list()

for img in X_test:
    xtest_list.append(ndimage.rotate(
        img,
        angle=np.random.uniform(0, 360),
        reshape=False,
        mode="constant",
        cval=0.,
        prefilter=False
    ))


mnist_images_train = np.round(np.array(xtrain_list),0) / 255.
mnist_images_train = mnist_images_train.clip(min=0., max=1.0)
mnist_images_test = np.round(np.array(xtest_list),0) / 255.
mnist_images_test = mnist_images_test.clip(min=0., max=1.0)


np.savez_compressed(destination_xtrain, mnist_images_train)
np.savez_compressed(destination_ytrain, mnist_label_train)
np.savez_compressed(destination_xtest, mnist_images_test)
np.savez_compressed(destination_ytest, mnist_label_test)