import abc

from typing import Optional, Sequence, cast

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
from gpflow.kernels.base import Kernel

from .gconvbase import GConvolutional


### Implementation of specific group-convolutional kernels where w_pg = w_p * w_g.
#
# One of the versions in chapter 5.2.2 "Weighted Group-Convolutional Kernel".



class D4Convolutional(GConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        patch_weights: Optional[TensorType] = None,
        groupelement_weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=8,
            colour_channels=colour_channels,
            patch_weights=patch_weights,
            groupelement_weights=groupelement_weights,
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
            [patches, patches90, patches180, patches270, patchesf, patches90f, patches180f, patches270f], axis=1
        ) # [num_data_x_num_patches_pretransform, ord(G), Wp, Hp, C]
        
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2] * self.grouporder, shp[3]]], 0)
        ) # "[batch..., N, num_patches_pretransform_x_ord(G), S]"
        return to_default_float(reshaped_patches)
    
    
    
    
    
    
class D2Convolutional(GConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        patch_weights: Optional[TensorType] = None,
        groupelement_weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200,
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
            patch_weights=patch_weights,
            groupelement_weights=groupelement_weights,
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
    
    
        
        
        
class RotationConvolutional(GConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        patch_weights: Optional[TensorType] = None,
        groupelement_weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200,
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
            patch_weights=patch_weights,
            groupelement_weights=groupelement_weights,
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
        
        
        
        
        
class Rotation180Convolutional(GConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        patch_weights: Optional[TensorType] = None,
        groupelement_weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=2,
            colour_channels=colour_channels,
            patch_weights=patch_weights,
            groupelement_weights=groupelement_weights,
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
        patchesr = tf.reverse(patches, axis=[-2, -3])
        patches = tf.stack(
            [patches, patchesr], axis=1
        ) # [num_data_x_num_patches_pretransform, ord(G), Wp, Hp, C]
        
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2] * self.grouporder, shp[3]]], 0)
        ) # "[batch..., N, num_patches_pretransform_x_ord(G), S]"
        return to_default_float(reshaped_patches)
    
    
        
        
        
class FlipLeftRightConvolutional(GConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        patch_weights: Optional[TensorType] = None,
        groupelement_weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=2,
            colour_channels=colour_channels,
            patch_weights=patch_weights,
            groupelement_weights=groupelement_weights,
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
        patchesf = tf.reverse(patches, axis=[-2])
        patches = tf.stack(
            [patches, patchesf], axis=1
        ) # [num_data_x_num_patches_pretransform, ord(G), Wp, Hp, C]
        
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2] * self.grouporder, shp[3]]], 0)
        ) # "[batch..., N, num_patches_pretransform_x_ord(G), S]"
        return to_default_float(reshaped_patches)
    
    
        
        
        
class FlipUpDownConvolutional(GConvolutional):
    r"""
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        patch_weights: Optional[TensorType] = None,
        groupelement_weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 200,
        max_parallel_iterations_kuf: int = 200,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=2,
            colour_channels=colour_channels,
            patch_weights=patch_weights,
            groupelement_weights=groupelement_weights,
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
        patchesf = tf.reverse(patches, axis=[-3])
        patches = tf.stack(
            [patches, patchesf], axis=1
        ) # [num_data_x_num_patches_pretransform, ord(G), Wp, Hp, C]
        
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2] * self.grouporder, shp[3]]], 0)
        ) # "[batch..., N, num_patches_pretransform_x_ord(G), S]"
        return to_default_float(reshaped_patches)