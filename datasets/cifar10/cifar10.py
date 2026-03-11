##### 
#
# Creates a local copy of converted data in numpy's npz format.
#
# Before running training and evaluation:
# Save raw cifar-10 dataset files in the same directory, change into that directory and run the following in command line:
# python cifar10.py
#
#####

import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

path_train = ["data_batch_" + str(i) for i in range(1, 6)]
path_test = "test_batch"

destination_xtrain = "cifar10/xtrain.npz"
destination_ytrain = "cifar10/ytrain.npz"
destination_xtest = "cifar10/xtest.npz"
destination_ytest = "cifar10/ytest.npz"

imported_shape = [10000, 3, 32, 32]
indices_transpose = [0, 2, 3, 1]
#new_shape = [10000, 32, 32, 3]

train_data_list = list()
train_label_list = list()
for batch in path_train:
    data_batch = unpickle(batch)
    images = np.transpose(np.reshape(data_batch[b"data"], imported_shape), indices_transpose).astype(np.float32) / 255.0
    train_data_list.append(images)
    train_label_list = train_label_list + data_batch[b'labels']
X_train = np.concat(train_data_list, axis=0)
Y_train = np.array(train_label_list)

data_batch = unpickle(path_test)
X_test = np.transpose(np.reshape(data_batch[b"data"], imported_shape), indices_transpose).astype(np.float32) / 255.0
Y_test = np.array(data_batch[b'labels'])

np.savez_compressed(destination_xtrain, X_train)
np.savez_compressed(destination_ytrain, Y_train)
np.savez_compressed(destination_xtest, X_test)
np.savez_compressed(destination_ytest, Y_test)