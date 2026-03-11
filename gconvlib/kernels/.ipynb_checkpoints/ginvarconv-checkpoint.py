import abc

from typing import Optional, Sequence, cast

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
from gpflow.kernels.base import Kernel

from .gconvbase import GInvarConvolutional


### Implementation of specific invariant group-convolutional kernels.
#
# Weights are mapped to space of fixed points of the group action by "averaging".
# That is done in the method get_weights.
# The method for weight-sharing is the one described in chapter 5.1.4 "Equivariant Weighted Kernel Maps"
# and chapter 5.2.2 "Weighted Group-Convolutional Kernel".
# Although it might not be that difficult to manually construct the weights from a lower-dimensional set
# of weights manually, for example in the case of rotation-invariance. 
#
# It is not a requirement for the group acting on the input images and the group acting on the weights to be
# the same.
# For example:
# The kernel map can be equivariant with respect to image rotations and the weights can be mapped to the space of
# fixed points under the action of the dihedral group D4.
# The resulting kernel would still be rotation-invariant because it is a subgroub of D4.
# The computational cost with respect to the number of required base kernel evaluations is lower
# while getting the benefits of weight-sharing, like lower intrinsic dimensionality of weights and 
# therefore being less prone to overfitting.
# The same for the other way around, e.g. if we want the group-convolutional GP be a bit more flexible.
# E.g. an equivariant "non-linearity" with respect to ninety degree image rotations and an 
# equivariant "linearity" with respect to 180 degree image rotations resulting in a convolutional GP
# that is invariant with respect to 180 degree rotations.


class D4InvarConvolutional(GInvarConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=8,
            colour_channels=colour_channels,
            weights=weights,
            strides=strides,
            max_parallel_iterations_kdiag = max_parallel_iterations_kdiag,
            max_parallel_iterations_kuf=max_parallel_iterations_kuf
        )
        
        
    def get_patches(self, X: TensorType) -> tf.Tensor:
        """ 
        """
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(
            X, 
            [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
        ) # "[num_data, W, H, C]"
        patches = tf.image.extract_patches(
            X,
            [1, self.patch_shape[0], self.patch_shape[1], 1],
            [1, self.strides[0], self.strides[1], 1],
            [1, 1, 1, 1],
            "VALID",
        ) #"[num_data, n_x_patches, n_y_patches, S]" where S=w_x_h_x_C
        shp = tf.shape(patches)
        
        patches = tf.reshape(
            patches, 
            [shp[0] * shp[1] * shp[2], self.patch_shape[0], self.patch_shape[1], self.colour_channels]
        ) # "[num_data_x_num_patches_pretransform, Wp, Hp, C]"
        
        # Create tensors of transformed patches
        # Naming convention: patchn stands for the rotation degree, patchf or patchnf for degree and then flip.
        # Ordering convention: 0, 90, 180, 270 counter-clockwise, then flip
        patches90 = tf.transpose(tf.reverse(patches, [-2]), [0, 2, 1, 3])
        patches180 = tf.reverse(patches, axis=[-2, -3])
        patches270 = tf.reverse(tf.transpose(patches, [0, 2, 1, 3]), [-2])
        patchesf = tf.reverse(patches, axis=[-2])
        patches90f = tf.reverse(patches90, axis=[-2])
        patches180f = tf.reverse(patches180, axis=[-2])
        patches270f = tf.reverse(patches270, axis=[-2])
        
        patches = tf.stack(
            [patches, patches90, patches180, patches270, patchesf, patches90f, patches180f, patches270f], 
            axis=1
        ) # [num_data_x_num_patches_pretransform, ord(G), Wp, Hp, C]
        
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2] * self.grouporder, shp[3]]], 0)
        ) # "[batch..., N, num_patches_pretransform_x_ord(G), S]"
        return to_default_float(reshaped_patches)
    
    @property
    def get_weights(self) -> TensorType:
        '''
        '''
        weights = tf.reshape(self.weights, [self.W_weights, self.H_weights])
        weights90 = tf.transpose(tf.reverse(weights, [-1]), [1, 0])
        weights180 = tf.reverse(weights, axis=[-1, -2])
        weights270 = tf.reverse(tf.transpose(weights, [1, 0]), [-1])
        weightsf = tf.reverse(weights, axis=[-1])
        weights90f = tf.reverse(weights90, axis=[-1])
        weights180f = tf.reverse(weights180, axis=[-1])
        weights270f = tf.reverse(weights270, axis=[-1])
        weights = tf.add_n(
            [weights, weights90, weights180, weights270, weightsf, weights90f, weights180f, weights270f]
        ) / self.grouporder
        weights = tf.reshape(weights, [self.num_patches_image])
        
        # Below dtype is specified as default_float from gpflow.config to ensure matching data types.
        ones = tf.ones((self.grouporder,), dtype=default_float())
        weights = weights[:, None] * ones[None, :] # "[P, ord(G)]"
        weights = tf.reshape(weights, [self.num_patches_image * self.grouporder])
        return weights
    
    
    
    
    
class D2InvarConvolutional(GInvarConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=4,
            colour_channels=colour_channels,
            weights=weights,
            strides=strides,
            max_parallel_iterations_kdiag = max_parallel_iterations_kdiag,
            max_parallel_iterations_kuf=max_parallel_iterations_kuf
        )
        
        
    def get_patches(self, X: TensorType) -> tf.Tensor:
        """
        """
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(
            X, 
            [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
        ) # "[num_data, W, H, C]"
        patches = tf.image.extract_patches(
            X,
            [1, self.patch_shape[0], self.patch_shape[1], 1],
            [1, self.strides[0], self.strides[1], 1],
            [1, 1, 1, 1],
            "VALID",
        ) #"[num_data, n_x_patches, n_y_patches, S]" where S=w_x_h_x_C
        shp = tf.shape(patches)
        
        patches = tf.reshape(
            patches, 
            [shp[0] * shp[1] * shp[2], self.patch_shape[0], self.patch_shape[1], self.colour_channels]
        ) # "[num_data_x_num_patches_pretransform, Wp, Hp, C]"
        
        # Create tensors of transformed patches
        # Naming convention: h for horizontal and v for vertical flip.
        # Ordering convention: first horizontal flip than vertical flips.
        patches_h = tf.reverse(patches, axis=[-2])
        patches_v = tf.reverse(patches, axis=[-3])
        patches_hv = tf.reverse(patches, axis=[-2, -3])
        patches = tf.stack(
            [patches, patches_h, patches_v, patches_hv], axis=1
        ) # [num_data_x_num_patches_pretransform, ord(G), Wp, Hp, C]
        
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2] * self.grouporder, shp[3]]], 0)
        ) # "[batch..., N, num_patches_pretransform_x_ord(G), S]"
        return to_default_float(reshaped_patches)
    
    @property
    def get_weights(self) -> TensorType:
        '''
        '''
        weights = tf.reshape(self.weights, [self.W_weights, self.H_weights])
        weights_h = tf.reverse(weights, axis=[-1])
        weights_v = tf.reverse(weights, axis=[-2])
        weights_hv = tf.reverse(weights, axis=[-1, -2])
        weights = tf.add_n(
            [weights, weights_h, weights_v, weights_hv]
        ) / self.grouporder
        weights = tf.reshape(weights, [self.num_patches_image])
        
        # Below dtype is specified as default_float from gpflow.config to ensure matching data types.
        ones = tf.ones((self.grouporder,), dtype=default_float())
        weights = weights[:, None] * ones[None, :] # "[P, ord(G)]"
        weights = tf.reshape(weights, [self.num_patches_image * self.grouporder])
        return weights
    
    
    
    
    
class RotationInvarConvolutional(GInvarConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 2000,
        max_parallel_iterations_kuf: int = 2000,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=4,
            colour_channels=colour_channels,
            weights=weights,
            strides=strides,
            max_parallel_iterations_kdiag = max_parallel_iterations_kdiag,
            max_parallel_iterations_kuf=max_parallel_iterations_kuf,
            normalisation_factor_scale=normalisation_factor_scale,
        )
        
    def get_patches(self, X: TensorType) -> tf.Tensor:
        """
        """
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(
            X, 
            [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
        ) # "[num_data, W, H, C]"
        patches = tf.image.extract_patches(
            X,
            [1, self.patch_shape[0], self.patch_shape[1], 1],
            [1, self.strides[0], self.strides[1], 1],
            [1, 1, 1, 1],
            "VALID",
        ) #"[num_data, n_x_patches, n_y_patches, S]" where S=w_x_h_x_C
        shp = tf.shape(patches)
        
        patches = tf.reshape(
            patches, 
            [shp[0] * shp[1] * shp[2], self.patch_shape[0], self.patch_shape[1], self.colour_channels]
        ) # "[num_data_x_num_patches_pretransform, Wp, Hp, C]"
        
        # Create tensors of transformed patches
        # Naming convention: patchn stands for the rotation degree.
        # Ordering convention: 0, 90, 180, 270 counter-clockwise.
        patches90 = tf.transpose(tf.reverse(patches, [-2]), [0, 2, 1, 3])
        patches180 = tf.reverse(patches, axis=[-2, -3])
        patches270 = tf.reverse(tf.transpose(patches, [0, 2, 1, 3]), [-2])
        patches = tf.stack(
            [patches, patches90, patches180, patches270], axis=1
        ) # [num_data_x_num_patches_pretransform, ord(G), Wp, Hp, C]
        
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2] * self.grouporder, shp[3]]], 0)
        ) # "[batch..., N, num_patches_pretransform_x_ord(G), S]"
        return to_default_float(reshaped_patches)
    
    @property
    def get_weights(self) -> TensorType:
        '''
        '''
        weights = tf.reshape(self.weights, [self.W_weights, self.H_weights])
        weights90 = tf.transpose(tf.reverse(weights, [-1]), [1, 0])
        weights180 = tf.reverse(weights, axis=[-1, -2])
        weights270 = tf.reverse(tf.transpose(weights, [1, 0]), [-1])
        weights = tf.add_n(
            [weights, weights90, weights180, weights270]
        ) / self.grouporder
        weights = tf.reshape(weights, [self.num_patches_image]) 
        
        # Below dtype is specified as default_float from gpflow.config to ensure matching data types.
        ones = tf.ones((self.grouporder,), dtype=default_float())
        weights = weights[:, None] * ones[None, :] # "[P, ord(G)]"
        weights = tf.reshape(weights, [self.num_patches_image * self.grouporder])
        
        return weights

# Implementation of other group-invariant convolutional kernels follow the same principle.