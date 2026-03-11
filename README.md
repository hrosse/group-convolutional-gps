# Group-Convolutional Gaussian Processes

Implementation of models that extend convolutional Gaussian processes to incorporate discrete
rotations and reflections.

## Background and Context

This repository contains the code developed as part of the master's thesis
"Group Equivariance in Convolutional Gaussian Processes".

The central question investigated in this work is whether convolutional
Gaussian processes (CGPs) [van der Wilk et al., 2017] can be extended to broader classes
of symmetries in order to generalize convolutional structure beyond
patch translations.

By analyzing the structure of a class of invariant covariance kernels with
sum structure [Ginsbourger et al., 2012] and studying their equivariance
properties, we obtain a natural extension of convolutional Gaussian processes
that incorporates discrete rotations and reflections.
In particular, the models considered here are based on group actions of the
dihedral group **D₄** and its subgroups, corresponding to the symmetries of a
square (90° rotations and reflections).

The resulting models — referred to as *group-convolutional Gaussian
processes* and *group-invariant convolutional Gaussian processes* —
are implemented using **GPflow**.

The repository further contains scripts for running a set of initial
proof-of-concept experiments used during the thesis.


## Overview of structure

- "gconvlib" is the core module containing implementations of kernels, interdomain covariance functions and such as well as a quick fix to GPflow's Softmax likelihood.
- There are scripts in "datasets" for preprocessing of data.
- Training and evaluation are separated.
- The python scripts named "mnist_rot.py" or similarly are used to start training with command-line arguments.
- Training scripts also contain a patch to multi-output inducing variables that disables shape checking. Without it GPflow does not allow inducing inputs that lie in spaces of different dimensionality as can be the case when latent GPs have different inter-domain inducing variables.
- Models (including intermediary results during training) are saved in "checkpoints".
- Running the python scripts with suffix "_eval.py" will load checkpoints, compute scores (e.g. ELBO, negative log predictive probability, classification error) and save them in "results".
- Evaluation results can be accessed and inspected by loading values, e.g. in a notebook. Example notebooks can be found in directory "results".



## Running Code for Experiments

Everything was implemented in GPflow version 2.9.2.

Experiments were run on a RTX 4090.

### Data Preparation

Copy the dataset into the corresponding "datasets" and run the corresponding scripts.

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