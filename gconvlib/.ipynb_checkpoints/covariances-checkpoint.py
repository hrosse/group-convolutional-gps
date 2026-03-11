from typing import Union

import numpy as np 
import tensorflow as tf

from gpflow.base import TensorLike, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive

import gpflow.covariances as covs

from .inducingvars import InducingImages, InducingPatchesLoopX
from gpflow.inducing_variables import InducingPatches

from .kernels.gkernels import GKernelBase
from.kernels.gconvbase import ConvolutionalBase





### Kuf ###


@covs.Kuf.register(InducingImages, GKernelBase, object)
def Kuf_gkernel_image(
    inducing_variable: InducingImages, kernel: GKernelBase, Xnew: TensorType
) -> tf.Tensor:
    '''
    Similar to Kuf of the convolutional GP in GPflow.
    Equivalent to Kuf_gconv_patch with a G-convolutional kernel when patch shape is equal to image shape.
    '''
    X_transformed = kernel.get_X_transformed(Xnew)  # [N, ord(G), D]
    bigKzx = kernel.base_kernel.K(
        inducing_variable.Z, X_transformed
    )  # [M, N, ord(G)] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(
        bigKzx * kernel.weights if hasattr(kernel, "weights") else bigKzx, [2]
    )
    return Kzx / kernel.grouporder





@covs.Kuf.register(InducingPatches, ConvolutionalBase, object)
def Kuf_gconv_patch(
    inducing_variable: InducingPatches, kernel: ConvolutionalBase, Xnew: TensorType
) -> tf.Tensor:
    '''
    Similar to the Kuf for the convolutional GP in GPflow.
    Here the weights are accessed with get_weights instead of directly accessing the attribute weights.
    The implementation of get_weights in the kernel class brings the weights into the right form, 
    e.g. mapping weights into the space of fixed points under the group action in case of invariant group convolutional GPs.
    The kernels have an attribute normalisation_factor_scale.
    Its purpose is to scale the normalisation factor in the kernel in case that the "usual" value is too large.
    Constraints on weights and parameter variance of the base kernel are necessary for stability.
    When values for weights or for base kernel parameter variance reach the bounds, the other parameters become increasingly large
    to compensate which might not be desirable.
    Chosing some appropriate value for normalisation_factor_scale can mitigate that effect.
    '''
    Xp = kernel.get_patches(Xnew)
    bigKzx = kernel.base_kernel.K(
        inducing_variable.Z, Xp
    ) 
    Kzx = tf.reduce_sum(bigKzx * kernel.get_weights, [2])
    return Kzx  / (to_default_float(kernel.num_patches) / kernel.normalisation_factor_scale)




# not used
@covs.Kuf.register(InducingPatchesLoopX, ConvolutionalBase, object)
def Kuf_conv_patchloopx(
    inducing_variable: InducingPatchesLoopX, kernel: ConvolutionalBase, Xnew: TensorType
) -> tf.Tensor:
    '''
    not used.
    See kernel class GConvolutionalBaseLoopDiag.
    '''
    return kernel.Kuf(inducing_variable.Z, Xnew)





### Kuu ###
#
# In all cases the covariances between inducing variables are given by just evaluating the base kernel


@covs.Kuu.register(InducingImages, GKernelBase)
def Kuu_gkernel_image(
    inducing_variable: InducingImages, kernel: GKernelBase, jitter: float = 0.0
) -> tf.Tensor:
    '''
    '''
    return kernel.base_kernel.K(inducing_variable.Z) + jitter * tf.eye(
        inducing_variable.num_inducing, dtype=default_float()
    )





@covs.Kuu.register(InducingPatches, ConvolutionalBase)
def Kuu_gconv_patch(
    inducing_variable: InducingPatches, kernel: ConvolutionalBase, jitter: float = 0.0
) -> tf.Tensor:
    '''
    '''
    return kernel.base_kernel.K(inducing_variable.Z) + jitter * tf.eye(
        inducing_variable.num_inducing, dtype=default_float()
    )




# not used
@covs.Kuu.register(InducingPatchesLoopX, ConvolutionalBase)
def Kuu_conv_patchloopx(
    inducing_variable: InducingPatchesLoopX, kernel: ConvolutionalBase, jitter: float = 0.0
) -> tf.Tensor:
    '''
    not used
    '''
    return kernel.base_kernel.K(inducing_variable.Z) + jitter * tf.eye(
        inducing_variable.num_inducing, dtype=default_float()
    )
