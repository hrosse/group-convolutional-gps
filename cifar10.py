"""
===========================================================
 Experiment CIFAR-10
===========================================================

This script provides pipelines for the experiments with the CIFAR-10 dataset.

Flow:
---------
1. Parse arguments, stored in file arg_dict.json.
2. Log environment versions (TensorFlow, GPflow, TensorFlow Probability).
3. Load training/test data via function load_data.
4. Build GP models with function build_model.
5. Training with Adam optimiser, checkpointing and logging elapsed time.
6. Stop early if NaNs are detected or the maximum time is exceeded.

Outputs:
-------------
- arg_dict.json: experiment configuration.
- version_log.json: library versions used.
- time_log.json: training steps and elapsed time.
- Checkpoints: saved in the specified directory save_dir.

Structure:
----------
- parse_args: Parse command line arguments.
- load_data: Load dataset.
- build_model: Construct GP model.
- get_optimizer: Configure ADAM optimiser.
- has_bad_params: Check for NaNs/Infs in model parameters.
- train_and_save: Run training loop and save checkpoints and logs.
- log_versions: Store used version info.

===========================================================
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.config import default_float
from gpflow.utilities import to_default_float, to_default_int

from gconvlib.models import SVGP_with_min_var

from gpflow.kernels import SquaredExponential, Sum, Convolutional, LinearCoregionalization
from gconvlib.kernels.sharedlscalekernels import ACRDKernel, AFCPRDKernel
from gconvlib.kernels.plainconv import FullConvolutional
from gconvlib.kernels.gkernels import D2Kernel, D4Kernel
from gconvlib.kernels.gconv import D2Convolutional, FlipLeftRightConvolutional
from gconvlib.kernels.gconvfullweights import D2ConvolutionalFullWeights, FlipLeftRightConvolutionalFullWeights

from gpflow.inducing_variables import InducingPatches, InducingPoints, SeparateIndependentInducingVariables, FallbackSeparateIndependentInducingVariables
from gconvlib.inducingvars import InducingImages 

from gconvlib.likelihoods import SoftmaxFixed
from gpflow.likelihoods import MultiClass, RobustMax, Gaussian

# The line below results in registration of custom Kuf and Kuu instances
import gconvlib.covariances

# Disable too strict shape checking.
# Inducing points can have different dimensionality.
FallbackSeparateIndependentInducingVariables.__init__ = FallbackSeparateIndependentInducingVariables.__init__.__wrapped__



def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPflow SVGP model")

    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    parser.add_argument("--rseed", type=int, default=101,
                        help="Random seed for reproducibility")

    # Directory paths
    parser.add_argument("--data_dir", type=str, default="./datasets/cifar10/cifar10",
                        help="Directory containing dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/cifar10",
                        help="Directory to save checkpoints and logs")

    # Default float options and default jitter
    parser.add_argument("--default_dtype", type=str, choices=["f32", "f64", "f16"], default="f32",
                        help="Default dtype (maps to np.float32, np.float64, np.float16)")
    parser.add_argument("--default_jitter", type=float, default=1e-4,
                        help="Default jitter value") # Rather large because apparently numerical issues with APRD and ACRD (Chol.dec. fail)

    # Model options, inducing points number and sampling options
    parser.add_argument("--model", type=str,
                        choices=["SE", "caddAPRD", "D4caddAPRD", "caddConvSE", "FlipConvFullWandD4caddAPRD", "caddD2ConvFWp6s2andD4caddAPRD"],
                        required=True,
                        help="Model type")
    parser.add_argument("--M", type=int, required=True,
                        help="Number of inducing points")
    parser.add_argument("--p_sampling", type=str, choices=["data", "uniform", "mixed"], default="uniform",
                        help="Sample strategy for inducing patches")
    parser.add_argument("--im_sampling", type=str, choices=["data", "uniform", "mixed"], default="data",
                        help="Sample strategy for inducing images")
    parser.add_argument("--likelihood", type=str, choices=["softmax", "robustmax", "gaussian"], default="softmax",
                        help="Likelihood for the model")

    ### Optimisation options

    parser.add_argument("--mb_size", type=int, required=True,
                        help="Minibatch size")

    # Stopping Conditions for training
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum number of training steps")
    parser.add_argument("--max_time", type=float, default=None,
                        help="Maximum training time in seconds")

    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save checkpoint every N steps")

    # Learning rate and learning rate schedule
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--lr_decay_steps", type=int, default=None,
                        help="Steps after which to reduce learning rate")
    parser.add_argument("--lr_decay_factor", type=float, default=None,
                        help="Factor with which to multiply learning rate")

    args = parser.parse_args()

    # Map dtype string to numpy dtype
    dtype_map = {"f32": np.float32, "f64": np.float64, "f16": np.float16}
    args.default_dtype = dtype_map[args.default_dtype]

    args = vars(args)

    # Separate directory for each selected model in specified checkpoint directory
    args["save_dir"] = args["save_dir"] + "/" + args["model"]

    return args


def load_data(arg_dict):
    """
    Returns (Xtrain, Ytrain, Xtest, Ytest, arg_dict).
    """
    xtrain_path = os.path.join(arg_dict["data_dir"], "xtrain.npz")
    if not os.path.exists(xtrain_path):
        raise FileNotFoundError(f"No dataset found at {xtrain_path}")

    ytrain_path = os.path.join(arg_dict["data_dir"], "ytrain.npz")
    if not os.path.exists(ytrain_path):
        raise FileNotFoundError(f"No dataset found at {ytrain_path}")

    xtest_path = os.path.join(arg_dict["data_dir"], "xtest.npz")
    if not os.path.exists(xtest_path):
        raise FileNotFoundError(f"No dataset found at {xtest_path}")

    ytest_path = os.path.join(arg_dict["data_dir"], "ytest.npz")
    if not os.path.exists(ytest_path):
        raise FileNotFoundError(f"No dataset found at {ytest_path}")

    with np.load(xtrain_path) as data:
        Xtrain = data["arr_0"]
    with np.load(ytrain_path) as data:
        Ytrain = data["arr_0"]
    with np.load(xtest_path) as data:
        Xtest = data["arr_0"]
    with np.load(ytest_path) as data:
        Ytest = data["arr_0"]

    num_data = Xtrain.shape[0] # Number of points in training data.
    image_shape = Xtrain.shape[1:-1]
    colour_channels = Xtrain.shape[-1]
    num_classes = np.unique(Ytrain).shape[0]

    # Flatten images
    Xtrain = np.reshape(Xtrain, [Xtrain.shape[0], np.prod(Xtrain.shape[1:])])
    Xtest = np.reshape(Xtest, [Xtest.shape[0], np.prod(Xtest.shape[1:])])

    # Required arguments when building models, depend on the data
    arg_dict["num_data"] = num_data
    arg_dict["num_classes"] = num_classes
    arg_dict["image_shape"] = image_shape
    arg_dict["colour_channels"] = colour_channels

    # Encoding of labels as required by likelihood
    if arg_dict["likelihood"] == "softmax":
        Ytrain_hot = np.zeros((num_data, num_classes), dtype=int)
        Ytrain_hot[np.arange(num_data), Ytrain.astype("int")] = 1.0
        Ytrain = to_default_int(Ytrain_hot)
        Ytest_hot = np.zeros((Xtest.shape[0], num_classes), dtype=int)
        Ytest_hot[np.arange(Xtest.shape[0]), Ytest.astype("int")] = 1.0
        Ytest = to_default_int(Ytest_hot)
    elif arg_dict["likelihood"] == "robustmax":
        Ytrain = to_default_int(Ytrain)
        Ytest = to_default_int(Ytest)
    elif arg_dict["likelihood"] == "gaussian":
        Ytrain_hot = np.zeros((num_data, num_classes), dtype=int)
        Ytrain_hot[np.arange(num_data), Ytrain.astype("int")] = 1.0
        Ytrain = to_default_float(Ytrain_hot)
        Ytest_hot = np.zeros((Xtest.shape[0], num_classes), dtype=int)
        Ytest_hot[np.arange(Xtest.shape[0]), Ytest.astype("int")] = 1.0
        Ytest = to_default_float(Ytest_hot)
        
    # Cast data to default float, returns tensors
    Xtrain = to_default_float(Xtrain)
    Xtest = to_default_float(Xtest)

    return Xtrain, Ytrain, Xtest, Ytest, arg_dict



def build_model(Xtrain, arg_dict):
    # Block below is used as standard values in experiments.
    # Values might be replaced in the code blocks where specific models are build. 
    var_init_image = 1.0
    var_init_patch = 1.0
    lscale_init_image = 5.0
    lscale_init_patch = 1.0
    patch_shape_aprd = [2, 2]
    patch_shape_conv = [5, 5]
    patch_shape_gconv = [5, 5]
    strides_conv = [1, 1]
    strides_gconv = [1, 1]

    if arg_dict["likelihood"] == "softmax":
        likelihood = SoftmaxFixed(arg_dict["num_classes"])
    elif arg_dict["likelihood"] == "robustmax":
        # encountered issue when default float is not float64
        invlink = RobustMax(arg_dict["num_classes"])
        likelihood = MultiClass(arg_dict["num_classes"], invlink=invlink)
    elif arg_dict["likelihood"] == "gaussian":
        likelihood = Gaussian()

    # Below are some convenient functions that return bijectors tailored to specific models/kernels.
    # Define constraints on kernel hyperparameters.
    def affine_scalar_bijector(shift=None, scale=None):
        scale_bijector = (
            tfp.bijectors.Scale(scale) if scale else tfp.bijectors.Identity()
        )
        shift_bijector = (
            tfp.bijectors.Shift(shift) if shift else tfp.bijectors.Identity()
        )
        return shift_bijector(scale_bijector)
    ftype = lambda x: np.array(x, dtype=default_float())

    # Patch weights in FullConvolutional and GConvolutionalFullWeights
    max_abs_w_patch = lambda: affine_scalar_bijector(shift=ftype(-25.0), scale=ftype(50.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # Group element weights in GKernel
    max_abs_w_imtransform = lambda: affine_scalar_bijector(shift=ftype(-10.0), scale=ftype(20.0))(
        tfp.bijectors.Sigmoid()
    ) 
    ### In GConvolutional kernel the "patch weights" are given by the product of "location weights" and "transformation weights" 
    # Patch location weights in GConvolutional
    max_abs_w_patchlocation = lambda: affine_scalar_bijector(shift=ftype(-7.0), scale=ftype(14.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # Group element weights in GConvolutional
    max_abs_w_patchtransform = lambda: affine_scalar_bijector(shift=ftype(-3.0), scale=ftype(6.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # Upper/lower bound on variance of kernel. When there are no additional weights as in GKernel or variants of GConvKernel
    var_constrained = lambda: affine_scalar_bijector(shift=ftype(1e-4), scale=ftype(80.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # Upper/lower bound on variance of base kernel, "basis function weights" are effectively the product of variance and weights 
    var_base_constrained = lambda: affine_scalar_bijector(shift=ftype(1e-4), scale=ftype(30.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # For SE, ACRD or APRD (base) kernels, i.e. when there is no combination of APRD and ACRD
    ls_constrained = lambda: affine_scalar_bijector(shift=ftype(1e-4), scale=ftype(50.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # For ACRD base kernel of patch response function, Cholesky decomposition failed, maybe here is the issue?
    ls_patch_acrd_constrained = lambda: affine_scalar_bijector(shift=ftype(1e-4), scale=ftype(9.0))(
        tfp.bijectors.Sigmoid()
    ) 
    # For kernels that combine APRD and ACRD. 
    # Inputs are scaled in succession by patch location weights and colour channel weights (i.e. the product).
    # Upper bound effectively is squared scale.
    ls_cadd_aprd_constrained = lambda: affine_scalar_bijector(shift=ftype(1e-4), scale=ftype(7.0))(
        tfp.bijectors.Sigmoid()
    ) 

    def sample_inducing_images(Xtrain, arg_dict, kernel=None, r_uniform=0.5):
        if arg_dict["im_sampling"] == "data":
            mask = np.random.choice(range(Xtrain.shape[0]), arg_dict["M"], replace=False)
            Z = Xtrain.numpy()[mask]
        elif arg_dict["im_sampling"] == "uniform":
            Z = np.random.randint(0, 255, size=(arg_dict["M"], Xtrain.shape[-1]))/255
        elif arg_dict["im_sampling"] == "mixed":
            Z_uniform = np.random.randint(0, 255, size=(arg_dict["M"], Xtrain.shape[-1]))/255
            mask = np.random.choice(range(Xtrain.shape[0]), arg_dict["M"], replace=False)
            Z_data = Xtrain.numpy()[mask]
            Z = r_uniform * Z_uniform + (1 - r_uniform) * Z_data
        return InducingImages(Z)

    def sample_inducing_patches(Xtrain, arg_dict, kernel, r_uniform=0.1):
        if arg_dict["p_sampling"] == "uniform":
            patch_len = int(np.prod([np.prod(kernel.patch_shape), kernel.colour_channels]))
            Z = np.random.randint(0, 255, size=(arg_dict["M"], patch_len))/255
        elif arg_dict["p_sampling"] == "data":
            # First sample M points from data
            mask = np.random.choice(range(Xtrain.shape[0]), arg_dict["M"], replace=False)
            Z = Xtrain.numpy()[mask]
            # Sample patch from each of the M points
            Z = kernel.get_patches(Z).numpy()
            Z_list = list()
            for i in range(Z.shape[0]):
                Z_list.append(Z[i][np.random.randint(0, kernel.num_patches)])
            #Z_unique = np.unique(Z_list, axis=0)   # In case duplicates should be removed
            Z = np.array(Z_list)
        elif arg_dict["p_sampling"] == "mixed":
            patch_len = int(np.prod([np.prod(kernel.patch_shape), kernel.colour_channels]))
            Z_uniform = np.random.randint(0, 255, size=(arg_dict["M"], patch_len))/255
            # First sample M points from data
            mask = np.random.choice(range(Xtrain.shape[0]), arg_dict["M"], replace=False)
            Z_data = Xtrain.numpy()[mask]
            # Sample patch from each of the M points
            Z_data = kernel.get_patches(Z_data).numpy()
            Z_list = list()
            for i in range(Z_data.shape[0]):
                Z_list.append(Z_data[i][np.random.randint(0, kernel.num_patches)])
            Z_data = np.array(Z_list)
            Z = r_uniform * Z_uniform + (1 - r_uniform) * Z_data
        return InducingPatches(Z)

    def initialise_SVGP_model(kernel, likelihood, Z, arg_dict):
        model = SVGP_with_min_var(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=Z,
            num_data=arg_dict["num_data"],
            num_latent_gps=arg_dict["num_latent"]
        )
        return model

    def get_channel_additive_kernel(image_shape, colour_channels, var_init_base_kernel=1.0, with_first_order=False, with_last_order=False, ls_transform=None, var_transform=None):
        if ls_transform is None:
            ls_transform = lambda: gpflow.utilities.positive()
        if var_transform is None:
            var_transform = lambda: gpflow.utilities.positive()
            
        kernel_list = list()
        
        if with_first_order:
            #D = np.prod(image_shape + [colour_channels])
            D = int(np.prod([np.prod(image_shape), colour_channels]))
            for i in range(colour_channels):
                active_dimensions = np.array(range(i, D, colour_channels))
                kernel = SquaredExponential(variance=var_init_base_kernel, active_dims=active_dimensions)
                kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_transform())
                kernel.variance = gpflow.Parameter(kernel.variance.numpy(), transform=var_transform())
                kernel_list.append(kernel)
                
        active_channels_list = [[0,1], [0,2], [1,2]] # RG, RB and GB
        for active_channels in active_channels_list:
            base_kernel = SquaredExponential(variance=var_init_base_kernel)
            base_kernel.variance = gpflow.Parameter(base_kernel.variance.numpy(), transform=var_transform())
            gpflow.set_trainable(base_kernel.lengthscales, False) # redundant since lengthscales are handled by ACRDKernel
            kernel = ACRDKernel(
                kernel_instance=base_kernel, image_shape=image_shape, 
                colour_channels=colour_channels, active_channels=active_channels
            )
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_transform())
            kernel_list.append(kernel)
            
        if with_last_order:
            base_kernel = SquaredExponential(variance=var_init_base_kernel)
            base_kernel.variance = gpflow.Parameter(base_kernel.variance.numpy(), transform=var_transform())
            gpflow.set_trainable(base_kernel.lengthscales, False) # redundant since lengthscales are handled by ACRDKernel
            kernel = ACRDKernel(kernel_instance=base_kernel, image_shape=image_shape, colour_channels=colour_channels)
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_transform())
            kernel_list.append(kernel)
        
        return Sum(kernel_list)


    def get_D2_channel_additive_kernel(image_shape, colour_channels, var_init_base_kernel=1.0, ls_transform=None, var_transform=None):
        if ls_transform is None:
            ls_transform = lambda: gpflow.utilities.positive()
        if var_transform is None:
            var_transform = lambda: gpflow.utilities.positive()
            
        kernel_list = list()
                
        active_channels_list = [[0,1], [1,2]] # RG and GB
        for active_channels in active_channels_list:
            base_kernel = SquaredExponential(variance=var_init_base_kernel)
            base_kernel.variance = gpflow.Parameter(base_kernel.variance.numpy(), transform=var_transform())
            gpflow.set_trainable(base_kernel.lengthscales, False) # redundant since lengthscales are handled by ACRDKernel
            kernel = ACRDKernel(
                kernel_instance=base_kernel, image_shape=image_shape, 
                colour_channels=colour_channels, active_channels=active_channels
            )
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_transform())
            kernel_list.append(kernel)
            
        # RGB
        base_kernel = SquaredExponential(variance=var_init_base_kernel)
        base_kernel.variance = gpflow.Parameter(base_kernel.variance.numpy(), transform=var_transform())
        gpflow.set_trainable(base_kernel.lengthscales, False) # redundant since lengthscales are handled by ACRDKernel
        kernel = ACRDKernel(kernel_instance=base_kernel, image_shape=image_shape, colour_channels=colour_channels)
        kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_transform())
        kernel_list.append(kernel)
        
        return Sum(kernel_list)




    if arg_dict["model"] == "SE":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]

        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            var = var_init_image
            l_scales = tf.ones(shape=(tf.shape(Xtrain)[-1],), dtype=default_float()) * lscale_init_image
            kernel = SquaredExponential(variance=var, lengthscales=l_scales)
            kernel.variance = gpflow.Parameter(kernel.variance.numpy(), transform=var_constrained())
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_constrained())
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))

        # Initialise multi-output kernel and multi-ouput inducing variables
        W = np.zeros((arg_dict["num_classes"], arg_dict["num_latent"] ))
        for i in range(W.shape[0]):
            W[i, i * latent_per_class : (i+1) * latent_per_class] = 1
        W = to_default_float(W)
        kernel = LinearCoregionalization(kernel_list, W=W)
        # Set mixing matrix W to non-trainable. Results in simply summing the right GPs
        gpflow.set_trainable(kernel.W, False)
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)


    

    elif arg_dict["model"] == "caddAPRD":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]

        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            # Initialise channel-additive AFCPRD kernel, sample inducing images
            base_kernel = get_channel_additive_kernel(
                image_shape=arg_dict["image_shape"], colour_channels=arg_dict["colour_channels"], 
                var_init_base_kernel=var_init_image, with_first_order=True, with_last_order=True,
                ls_transform=ls_cadd_aprd_constrained, var_transform=var_constrained
            )
            channel_additive_aprd_kernel = AFCPRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_aprd,
                colour_channels=arg_dict["colour_channels"]
            )
            channel_additive_aprd_kernel.lengthscales.assign(channel_additive_aprd_kernel.lengthscales.numpy() * lscale_init_image)
            kernel = channel_additive_aprd_kernel
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_cadd_aprd_constrained())
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))

        # Initialise multi-output kernel and multi-ouput inducing variables
        W = np.zeros((arg_dict["num_classes"], arg_dict["num_latent"] ))
        for i in range(W.shape[0]):
            W[i, i * latent_per_class : (i+1) * latent_per_class] = 1
        W = to_default_float(W)
        kernel = LinearCoregionalization(kernel_list, W=W)
        # Set mixing matrix W to non-trainable. Results in simply summing the right GPs
        gpflow.set_trainable(kernel.W, False)
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)




    elif arg_dict["model"] == "D4caddAPRD":
        latent_per_class = 1
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]

        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            # Initialise D4-channel-additive APRD kernel, sample inducing images
            base_kernel = get_channel_additive_kernel(
                image_shape=arg_dict["image_shape"], colour_channels=arg_dict["colour_channels"], 
                var_init_base_kernel=var_init_image, with_first_order=True, with_last_order=True,
                ls_transform=ls_cadd_aprd_constrained, var_transform=var_base_constrained
            )
            channel_additive_aprd_kernel = AFCPRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_aprd,
                colour_channels=arg_dict["colour_channels"]
            )
            channel_additive_aprd_kernel.lengthscales.assign(channel_additive_aprd_kernel.lengthscales.numpy() * lscale_init_image)
            channel_additive_aprd_kernel.lengthscales = gpflow.Parameter(channel_additive_aprd_kernel.lengthscales.numpy(), transform=ls_cadd_aprd_constrained())
            kernel = D4Kernel(
                base_kernel=channel_additive_aprd_kernel,
                image_shape=arg_dict["image_shape"],
                n_channel=arg_dict["colour_channels"]
            )
            kernel.weights = gpflow.Parameter(
                kernel.weights.numpy(),
                transform=max_abs_w_imtransform()
            )
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))

        # Initialise multi-output kernel and multi-ouput inducing variables
        W = np.zeros((arg_dict["num_classes"], arg_dict["num_latent"] ))
        for i in range(W.shape[0]):
            W[i, i * latent_per_class : (i+1) * latent_per_class] = 1
        W = to_default_float(W)
        kernel = LinearCoregionalization(kernel_list, W=W)
        # Set mixing matrix W to non-trainable. Results in simply summing the right GPs
        gpflow.set_trainable(kernel.W, False)
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)






    elif arg_dict["model"] == "caddConvSE":
        latent_per_class = 2
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]

        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            # Convolutional component
            var = var_init_patch
            l_scales = lscale_init_patch
            base_kernel = get_channel_additive_kernel(
                image_shape=patch_shape_conv, colour_channels=arg_dict["colour_channels"], 
                var_init_base_kernel=var_init_patch, with_first_order=False, with_last_order=False,
                ls_transform=ls_patch_acrd_constrained, var_transform=var_base_constrained
            )
            kernel = FullConvolutional(
                base_kernel=base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_conv,
                colour_channels=arg_dict["colour_channels"],
                strides=strides_conv,
                normalisation_factor_scale=2
            )
            kernel.weights = gpflow.Parameter(
                kernel.weights.numpy(),
                transform=max_abs_w_patch()
            )
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_patches(Xtrain, arg_dict, kernel))
            
            # Initialise channel-additive AFCPRD kernel, sample inducing images
            base_kernel = get_channel_additive_kernel(
                image_shape=arg_dict["image_shape"], colour_channels=arg_dict["colour_channels"], 
                var_init_base_kernel=var_init_image, with_first_order=True, with_last_order=True,
                ls_transform=ls_cadd_aprd_constrained, var_transform=var_constrained
            )
            channel_additive_aprd_kernel = AFCPRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_aprd,
                colour_channels=arg_dict["colour_channels"]
            )
            channel_additive_aprd_kernel.lengthscales.assign(channel_additive_aprd_kernel.lengthscales.numpy() * lscale_init_image)
            kernel = channel_additive_aprd_kernel
            kernel.lengthscales = gpflow.Parameter(kernel.lengthscales.numpy(), transform=ls_cadd_aprd_constrained())
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))

        # Initialise multi-output kernel and multi-ouput inducing variables
        W = np.zeros((arg_dict["num_classes"], arg_dict["num_latent"] ))
        for i in range(W.shape[0]):
            W[i, i * latent_per_class : (i+1) * latent_per_class] = 1
        W = to_default_float(W)
        kernel = LinearCoregionalization(kernel_list, W=W)
        # Set mixing matrix W to non-trainable. Results in simply summing the right GPs
        gpflow.set_trainable(kernel.W, False)
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)




    elif arg_dict["model"] == "FlipConvFullWandD4caddAPRD":
        #latent_per_class = 3
        latent_per_class = 2
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]

        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            # Initialise convolutional kernel, sample inducing patches
            var = var_init_patch
            l_scales = lscale_init_patch
            acrd_base_kernel = SquaredExponential(variance=var)
            acrd_base_kernel.variance = gpflow.Parameter(acrd_base_kernel.variance.numpy(), transform=var_base_constrained())
            gpflow.set_trainable(acrd_base_kernel.lengthscales, False) # redundant since lengthscales are handled by ACRDKernel
            base_kernel = ACRDKernel(kernel_instance=acrd_base_kernel, image_shape=patch_shape_gconv, colour_channels=arg_dict["colour_channels"])
            base_kernel.lengthscales = gpflow.Parameter(base_kernel.lengthscales.numpy(), transform=ls_patch_acrd_constrained())
            kernel = FlipLeftRightConvolutionalFullWeights(
                base_kernel=base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_gconv,
                colour_channels=arg_dict["colour_channels"],
                strides=strides_gconv,
                normalisation_factor_scale=4 # 2*grouporder
            )
            kernel.weights = gpflow.Parameter(
                kernel.weights.numpy(),
                transform=max_abs_w_patch()
            )
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_patches(Xtrain, arg_dict, kernel))
            
            # Initialise D4-channel-additive APRD kernel, sample inducing images
            base_kernel = get_channel_additive_kernel(
                image_shape=arg_dict["image_shape"], colour_channels=arg_dict["colour_channels"], 
                var_init_base_kernel=var_init_image, with_first_order=False, with_last_order=True,
                ls_transform=ls_cadd_aprd_constrained, var_transform=var_base_constrained
            )
            channel_additive_aprd_kernel = AFCPRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_aprd,
                colour_channels=arg_dict["colour_channels"]
            )
            channel_additive_aprd_kernel.lengthscales.assign(channel_additive_aprd_kernel.lengthscales.numpy() * lscale_init_image)
            channel_additive_aprd_kernel.lengthscales = gpflow.Parameter(channel_additive_aprd_kernel.lengthscales.numpy(), transform=ls_cadd_aprd_constrained())
            kernel = D4Kernel(
                base_kernel=channel_additive_aprd_kernel,
                image_shape=arg_dict["image_shape"],
                n_channel=arg_dict["colour_channels"]
            )
            kernel.weights = gpflow.Parameter(
                kernel.weights.numpy(),
                transform=max_abs_w_imtransform()
            )
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))

        # Initialise multi-output kernel and multi-ouput inducing variables
        W = np.zeros((arg_dict["num_classes"], arg_dict["num_latent"] ))
        for i in range(W.shape[0]):
            W[i, i * latent_per_class : (i+1) * latent_per_class] = 1
        W = to_default_float(W)
        kernel = LinearCoregionalization(kernel_list, W=W)
        # Set mixing matrix W to non-trainable. Results in simply summing the right GPs
        gpflow.set_trainable(kernel.W, False)
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)



    elif arg_dict["model"] == "caddD2ConvFWp6s2andD4caddAPRD":
        patch_shape_gconv = [6, 6]
        strides_gconv = [2, 2]
        
        latent_per_class = 2
        arg_dict["num_latent"] = latent_per_class * arg_dict["num_classes"]

        # Initialise kernel, sample and initialise inducing points
        kernel_list = list()
        Z_list = list()
        for i in range(arg_dict["num_classes"]):
            # Initialise convolutional kernel, sample inducing patches
            var = var_init_patch
            l_scales = lscale_init_patch
            base_kernel = get_D2_channel_additive_kernel(
                image_shape=patch_shape_gconv, colour_channels=arg_dict["colour_channels"], 
                var_init_base_kernel=var_init_patch,
                ls_transform=ls_patch_acrd_constrained, var_transform=var_base_constrained
            )
            kernel = D2ConvolutionalFullWeights(
                base_kernel=base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_gconv,
                colour_channels=arg_dict["colour_channels"],
                strides=strides_gconv,
                normalisation_factor_scale=4 #
            )
            kernel.weights = gpflow.Parameter(
                kernel.weights.numpy(),
                transform=max_abs_w_patch()
            )
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_patches(Xtrain, arg_dict, kernel))
            
            # Initialise D4-channel-additive APRD kernel, sample inducing images
            base_kernel = get_channel_additive_kernel(
                image_shape=arg_dict["image_shape"], colour_channels=arg_dict["colour_channels"], 
                var_init_base_kernel=var_init_image, with_first_order=False, with_last_order=True,
                ls_transform=ls_cadd_aprd_constrained, var_transform=var_base_constrained
            )
            channel_additive_aprd_kernel = AFCPRDKernel(
                base_kernel,
                image_shape=arg_dict["image_shape"],
                patch_shape=patch_shape_aprd,
                colour_channels=arg_dict["colour_channels"]
            )
            channel_additive_aprd_kernel.lengthscales.assign(channel_additive_aprd_kernel.lengthscales.numpy() * lscale_init_image)
            channel_additive_aprd_kernel.lengthscales = gpflow.Parameter(channel_additive_aprd_kernel.lengthscales.numpy(), transform=ls_cadd_aprd_constrained())
            kernel = D4Kernel(
                base_kernel=channel_additive_aprd_kernel,
                image_shape=arg_dict["image_shape"],
                n_channel=arg_dict["colour_channels"]
            )
            kernel.weights = gpflow.Parameter(
                kernel.weights.numpy(),
                transform=max_abs_w_imtransform()
            )
            kernel_list.append(kernel)
            Z_list.append(sample_inducing_images(Xtrain, arg_dict, kernel))

        # Initialise multi-output kernel and multi-ouput inducing variables
        W = np.zeros((arg_dict["num_classes"], arg_dict["num_latent"] ))
        for i in range(W.shape[0]):
            W[i, i * latent_per_class : (i+1) * latent_per_class] = 1
        W = to_default_float(W)
        kernel = LinearCoregionalization(kernel_list, W=W)
        # Set mixing matrix W to non-trainable. Results in simply summing the right GPs
        gpflow.set_trainable(kernel.W, False)
        Z = SeparateIndependentInducingVariables(Z_list)
        model = initialise_SVGP_model(kernel, likelihood, Z, arg_dict)

        
    
    return model, arg_dict
    


def get_optimizer(arg_dict):
    """
    Return an Adam optimizer with optional exponential learning rate decay.
    """
    lr = arg_dict["lr"]
    if arg_dict["lr_decay_steps"] > 0:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=arg_dict["lr"],
            decay_steps=arg_dict["lr_decay_steps"],
            decay_rate=arg_dict["lr_decay_factor"],
            staircase=True,
        )
    return tf.keras.optimizers.Adam(learning_rate=lr)


def has_bad_params(model):
    '''
    Model parameters become Nan when Cholesky decomposition fails.
    Also frequently occurs when variance is not clipped to a small minimum value
    when using SVGP.
    '''
    for var in model.trainable_variables:
        if not np.all(np.isfinite(var.numpy())):
            return True
    return False


def train_and_save(model, data, arg_dict):

    # Iterator over minibatches
    train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(model.num_data)
    train_iter = iter(train_dataset.batch(arg_dict["mb_size"]))
    
    training_loss = model.training_loss_closure(train_iter, compile=True)

    optimizer = get_optimizer(arg_dict)

    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            loss = training_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Setup directory for saving
    save_dir = arg_dict["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Setup checkpointing
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=None)

    # Save arg_dict
    with open(os.path.join(save_dir, "arg_dict.json"), "w") as f:
        json.dump(arg_dict, f, indent=4, default=str)

    # Save version log
    log_versions(save_dir)

    time_log = []
    start_time = time.time()

    nan_detected = False

    for step in range(1, arg_dict["max_steps"] + 1):
        optimization_step()

        if arg_dict["max_time"] is not None and (time.time() - start_time) > arg_dict["max_time"]:
            elapsed = time.time() - start_time
            ckpt_manager.save(checkpoint_number=step)
            time_log.append({"step": step, "elapsed_seconds": elapsed})
            with open(os.path.join(save_dir, "time_log.json"), "w") as f:
                    json.dump(time_log, f, indent=4)
            print(f"Max time {arg_dict['max_time']}s reached, stopping training.")
            print(f"Last step {step}: elapsed = {elapsed:.1f}s, checkpoint saved.")
            break

        if step % arg_dict["checkpoint_interval"] == 0:
            # Check for NaNs
            if has_bad_params(model):
                elapsed = time.time() - start_time
                time_log.append({"step": step, "elapsed_seconds": elapsed, "nan_detected": True})
                with open(os.path.join(save_dir, "time_log.json"), "w") as f:
                    json.dump(time_log, f, indent=4)
                print(f"NaN detected in model parameters at step {step}. Stopping training.")
                nan_detected = True
                break

            elapsed = time.time() - start_time
            ckpt_manager.save(checkpoint_number=step)
            time_log.append({"step": step, "elapsed_seconds": elapsed})
            if arg_dict["verbose"]:
                print(f"Step {step}: elapsed = {elapsed:.1f}s, checkpoint saved.")

            with open(os.path.join(save_dir, "time_log.json"), "w") as f:
                json.dump(time_log, f, indent=4)

    # Save timing log
    #with open(os.path.join(save_dir, "time_log.json"), "w") as f:
    #    json.dump(time_log, f, indent=4)

    if nan_detected:
        print(f"Training stopped early due to NaNs. Logs saved in {save_dir}")
    else:
        print(f"Training finished. Logs saved in {save_dir}")



def log_versions(save_dir):
    versions = {
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "gpflow": gpflow.__version__,
    }
    with open(os.path.join(save_dir, "version_log.json"), "w") as f:
        json.dump(versions, f, indent=4)


def main():
    arg_dict = parse_args()

    # Set default float and jitter
    gpflow.config.set_default_float(arg_dict["default_dtype"])
    gpflow.config.set_default_jitter(arg_dict["default_jitter"])

    # Set random seed
    np.random.seed(arg_dict["rseed"])
    tf.random.set_seed(arg_dict["rseed"])

    # Load data
    Xtrain, Ytrain, Xtest, Ytest, arg_dict = load_data(arg_dict)

    # Build model
    model, arg_dict = build_model(Xtrain, arg_dict)

    # Train + save
    train_and_save(model, (Xtrain, Ytrain), arg_dict)

if __name__ == "__main__":
    main()
