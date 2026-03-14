# Group-Convolutional Gaussian Processes

Implementation of models that extend convolutional Gaussian processes to incorporate discrete
rotations and reflections.

## Background and Context

This repository contains the code developed as part of the master's thesis
"Group Equivariance in Convolutional Gaussian Processes".

The central question investigated in this work is whether convolutional
Gaussian processes [van der Wilk et al., 2017] can be extended to broader classes
of symmetries in order to generalise convolutional structure beyond
patch translations.

By analysing the structure of a class of invariant covariance kernels with
sum structure [Ginsbourger et al., 2012] and studying their equivariance
properties, we obtain a natural extension of convolutional Gaussian processes
that incorporates discrete rotations and reflections.
In particular, the models considered here are based on group actions of the
dihedral group **D₄** and its subgroups.

The resulting models — referred to as *group-convolutional Gaussian
processes* and *group-invariant convolutional Gaussian processes* —
are implemented using **GPflow**.

The repository further contains scripts for running a set of initial
proof-of-concept experiments used during the thesis.


## Overview of Structure and Contents

Implementation of kernels, interdomain covariance functions and a few necessary modifications are found in *gconvlib*.

Training and evaluation are separated.
During training model parameters and the initialisation arguments are stored in *checkpoints*.
The evaluation scripts load the checkpoints, compute the evidence lower bound, classification error and negative log predictive probability.
These are saved in *results* from where they can be accessed.

Directory *etc* includes a notebook illustrating how constraints on the weights of a double-sum kernel results in an invariant kernel.
The construction of group-invariant convolutional GPs follows the same principle.

Implementation focussed on correctness rather than efficiency.
Thus large parts of the code are far from optimal and leave a lot of room for improvement.
For example, during training of the rotation-invariant convolutional GP models, GPU utilisation peaked somewhere around fifty percent.
The experiments should just be seen as a first proof-of-concept.
Due to time constraints things like reasonable appearing values for learning rate, learning rate decay and decay steps were "guessed".

## Running Experiments and Setup

Experiments were run on a RTX 4090.
The GPflow version was 2.9.2.

### Pre-processing Datasets

Copy the unpacked files of the raw dataset into the corresponding directory in "datasets", e.g. into "datasets/mnist".
Change into the directory and run the corresponding scripts, e.g. to create the rotated version of MNIST:

```console
python mnist_rot.py
```

The processed data is then written to "datasets/mnist/mnist_rot" in npz format.
The files are named "xtrain.npz", "ytrain.npz", "xtest.npz" and "ytest.npz".

### Rotated MNIST 0-vs-1

```console

python mnistrot01.py --model SE --M 125 --mb_size 150 --checkpoint_interval 2000 --max_steps 160000 --max_time 36000 --lr 0.01 --lr_decay_steps 25000 --lr_decay_factor 0.45 --verbose

python mnistrot01.py --model Rinv --M 125 --mb_size 150 --checkpoint_interval 2000 --max_steps 160000 --max_time 36000 --lr 0.01 --lr_decay_steps 25000 --lr_decay_factor 0.45 --verbose

python mnistrot01.py --model Conv --M 125 --mb_size 100 --checkpoint_interval 2000 --max_steps 160000 --max_time 48000 --lr 0.01 --lr_decay_steps 25000 --lr_decay_factor 0.45 --verbose

python mnistrot01.py --model RinvConv --M 125 --mb_size 100 --checkpoint_interval 2000 --max_steps 160000 --max_time 48000 --lr 0.01 --lr_decay_steps 25000 --lr_decay_factor 0.45 --verbose

python mnistrot01.py --model RinvConvp8 --M 125 --mb_size 100 --checkpoint_interval 2000 --max_steps 160000 --max_time 48000 --lr 0.01 --lr_decay_steps 25000 --lr_decay_factor 0.45 --verbose
```

### Rotated MNIST 6-vs-9

```console

python mnistrot69.py --model SE --M 125 --mb_size 150 --checkpoint_interval 2000 --max_steps 350000 --max_time 36000 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose

python mnistrot69.py --model Rinv --M 125 --mb_size 150 --checkpoint_interval 2000 --max_steps 350000 --max_time 36000 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose

python mnistrot69.py --model Conv --M 125 --mb_size 100 --checkpoint_interval 2000 --max_steps 350000 --max_time 48000 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose

python mnistrot69.py --model RinvConv --M 125 --mb_size 100 --checkpoint_interval 2000 --max_steps 350000 --max_time 48000 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose

python mnistrot69.py --model RinvConvp8 --M 125 --mb_size 100 --checkpoint_interval 2000 --max_steps 350000 --max_time 48000 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose
```


### Rotated MNIST full


```console
python mnistrot.py --model SE --M 750 --mb_size 120 --checkpoint_interval 5000 --max_steps 350000 --max_time 172800 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose

python mnistrot.py --model Rinv --M 750 --mb_size 120 --checkpoint_interval 5000 --max_steps 350000 --max_time 172800 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose

python mnistrot.py --model ConvSE --M 750 --mb_size 100 --checkpoint_interval 5000 --max_steps 350000 --max_time 172800 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose

python mnistrot.py --model RinvConvandRinv --M 750 --mb_size 50 --checkpoint_interval 5000 --max_steps 350000 --max_time 259200 --lr 0.01 --lr_decay_steps 50000 --lr_decay_factor 0.45 --verbose
```

The 6-vs-9 experiment showed that all types of convolutional GPs performed worse than the model with a simple RBF kernel.
This indicates that the structure of rotated MNIST is generally difficult for GPs with convolutional structure.
Still, both rotation-invariant convolutional GPs notably outperformed the classical convolutional GP in the 6-vs-9 experiment.
The structure of rotated MNIST has the effect that the experiment results with the models chosen here are not that insightful.
Repeating the experiments with the same models as in the previous experiments might be better to compare the difference between the standard convolutional GP model and the rotation-invariant convolutional GP models.

### CIFAR-10

```console
python cifar10.py --model caddAPRD --M 1000 --mb_size 120 --checkpoint_interval 10000 --max_steps 500000 --max_time 288000 --lr 0.01 --lr_decay_steps 60000 --lr_decay_factor 0.5 --verbose

python cifar10.py --model D4caddAPRD --M 1000 --mb_size 120 --checkpoint_interval 10000 --max_steps 500000 --max_time 288000 --lr 0.01 --lr_decay_steps 60000 --lr_decay_factor 0.5 --verbose

python cifar10.py --model caddConvSE --M 1000 --p_sampling mixed --mb_size 32 --checkpoint_interval 10000 --max_steps 500000 --max_time 259200 --lr 0.01 --lr_decay_steps 60000 --lr_decay_factor 0.5 --verbose

python cifar10.py --model FlipConvFullWandD4caddAPRD --M 1000 --p_sampling mixed --mb_size 32 --checkpoint_interval 10000 --max_steps 500000 --max_time 259200 --lr 0.01 --lr_decay_steps 60000 --lr_decay_factor 0.5 --verbose

python cifar10.py --model caddD2ConvFWp6s2andD4caddAPRD --M 1000 --p_sampling mixed --mb_size 32 --checkpoint_interval 10000 --max_steps 500000 --max_time 259200 --lr 0.01 --lr_decay_steps 60000 --lr_decay_factor 0.5 --verbose
```


### Example for running experiment evaluations from terminal

```console
python3 mnistrot69_eval.py --save_dir ./checkpoints/mnist_rot69/RinvConv --batch_size 100 --verbose
```

## References

M. van der Wilk, C. E. Rasmussen, and J. Hensman, “Convolutional gaussian processes,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017.

D. Ginsbourger, X. Bay, O. Roustant, and L. Carraro, “Argumentwise invariant kernels for the approximation of invariant functions,” Annales de la Facult ́e de Sciences de Toulouse, vol. Tome 21, no. num ́ero 3, pp. p. 501–527, 2012