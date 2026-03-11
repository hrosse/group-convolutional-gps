from typing import Optional, Sequence, cast, List 

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
from gpflow.kernels.base import Kernel


# These are implementations of the kernels in chapter 6 "Hyperparameter-sharing Framework for RBF Kernels
# with image data".


class ACRDKernel(Kernel):
    '''
    Implements automatic channel relevance detection (ACRD).
    Dimensions belonging to the same channel are scaled by the same lengthscale.
    Equivalent to APRDKernel where patch_shape=image_shape.

    This is the version in chapter 6.3 "Automatic Channel Relevancy Detection".

    The channel-additive kernel in chapter 6.5 "Channel-additive Kernels" is constructed as
    a Sum kernel where each base kernel is an ACRDKernel.
    The attribute active_channels is used to  specify the channels of each base kernel.

    Specifying active channels is similar to the active dimensions in GPflow's kernel.
    If specified the argument is a list of channel indices.
    E.g. active_channels=[0,1] specifies that only the first to image channels are used in the kernel.
    The inputs are sliced before handing the sliced points to kernel_instance.
    '''

    def __init__(
        self,
        kernel_instance: Kernel,
        image_shape: Sequence[int],
        colour_channels: int,
        active_channels: Optional[Sequence[int]] = None,
        lengthscales: Optional[TensorType] = None,
    ) -> None:
        '''
        image_shape: [2]
        lengthscales: [C]
        '''
        super().__init__()
        self.image_shape = image_shape
        self.kernel_instance = kernel_instance
        self.colour_channels = colour_channels
        self.active_channels = active_channels
        self.n_active = self.colour_channels if self.active_channels is None else len(self.active_channels)
        self.lengthscales = Parameter(
            np.ones(self.n_active, dtype=default_float()) if lengthscales is None else lengthscales,
            transform=positive()
        )

    def slice_channels(self, X:TensorType) -> TensorType:
        '''
        X must have shape like [..., C]
        '''
        return tf.gather(params=X, indices=self.active_channels, axis=-1)

    def scale(self, X: TensorType) -> TensorType:
        '''
        X: [batch..., N, D]
        X_scaled: [batch..., N, L] where L is dimensionality of data after slicing
        '''
        if X is not None:
            input_shape = tf.shape(X)
            output_shape = tf.concat([input_shape[:-1], [self.n_active * self.image_shape[0] * self.image_shape[1]]], 0)
            bNxWxH = tf.reduce_prod(input_shape[:-1]) * self.image_shape[0] * self.image_shape[1]
            X_scaled = tf.reshape(X, [bNxWxH, self.colour_channels]) # shape [num_data_x_W_x_H, C]
            # Slice input when active channels are specified
            if self.active_channels is not None:
                X_scaled = self.slice_channels(X_scaled) # shape [num_data_x_W_x_H, L]
            X_scaled = X_scaled / self.lengthscales # usually sufficient
            X_scaled = tf.reshape(X_scaled, output_shape) # shape [batch..., N, L]
        else:
            X_scaled = X
        return X_scaled


    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        '''
        # if X2 is None equivalent to calling with only X specified
        return self.kernel_instance.K(self.scale(X), self.scale(X2))

    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        For (most) stationary kernels scaling this is not neccessary but included to keep implementation general.
        '''
        #return self.kernel_instance.K_diag(X)
        return self.kernel_instance.K_diag(self.scale(X))





class APRDKernel(Kernel):
    '''
    Implements a version automatic patch relevance detection (APRD).
    Dimensions belonging to the same patch are scaled by the same lengthscale.
    Partitions each channel into patches.
    When patches contain the values of all channels use AFCPRDKernel.

    Despite the name of the class THIS IS NOT the kernel in chapter 6.2 "Automatic Patch Relevancy Detection"
    and was not used in the experiments.    
    '''

    def __init__(
        self,
        kernel_instance: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int,
        lengthscales: Optional[TensorType] = None,
    ) -> None:
        '''
        image_shape: [2]
        patch_shape: [2]
        lengthscales: [C]
        '''
        super().__init__()
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.kernel_instance = kernel_instance
        self.colour_channels = colour_channels
        self.lengthscales = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if lengthscales is None else lengthscales,
            transform=positive()
        )

    def scale(self, X: TensorType) -> TensorType:
        '''
        X: [batch..., N, D]
        X_scaled: [batch..., N, D]
        '''
        if X is not None:
            input_shape = tf.shape(X)
            batch = tf.shape(X)[:-2]
            N = tf.shape(X)[-2]
            flat_batch = tf.reduce_prod(batch)
            num_data = flat_batch * N
            X = tf.reshape(
                X,
                [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
            ) # shape [num_data, W, H, 3]

            # if argument sizes and strides are equal it extracts disjoint patches
            Xp = tf.image.extract_patches(
                X,
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, 1, 1, 1],
                "VALID",
            ) # shape [num_data, n_x_patches, n_y_patches, w_x_h_x_C]
            shp = tf.shape(Xp)
            Xp = tf.reshape(
                Xp,
                [num_data, shp[1], shp[2], self.patch_shape[0], self.patch_shape[1], self.colour_channels]
            ) # shape [num_data, n_x_patches, n_y_patches, w, h, C]
            Xp = tf.transpose(
                Xp, [0, 5, 1, 2, 3, 4]
            ) # shape [num_data, C, n_x_patches, n_y_patches, w, h]
            Xp = tf.reshape(
                Xp,
                [num_data, self.colour_channels, shp[1], shp[2], tf.reduce_prod(self.patch_shape)]
            ) # shape [num_data, C, n_x_patches, n_y_patches, w_x_h]


            lengthscales_reshaped = tf.reshape(
                self.lengthscales, [1, self.colour_channels, shp[1], shp[2], 1]
            ) # shape [1, C, n_x_patches, n_y_patches, 1]
            lengthscales_reshaped = tf.repeat(
                lengthscales_reshaped, repeats=tf.reduce_prod(self.patch_shape), axis=-1
            ) # shape [1, C, n_x_patches, n_y_patches, w_x_h]

            # somehow dtype of lengthscales_reshaped have been changed. Thus to_default_float.
            # I will include to_default_float(Xp) just in case.
            #X_scaled = Xp / to_default_float(lengthscales_reshaped) # usually sufficient
            X_scaled = to_default_float(Xp) / to_default_float(lengthscales_reshaped)

            # transpose back so that input is in same order as output
            # relevant in case we have kernel instance that is combination of
            # kernels with active dimensions specified.
            X_scaled = tf.reshape(
                X_scaled,
                [num_data, self.colour_channels, shp[1], shp[2], self.patch_shape[0], self.patch_shape[1]]
            ) # shape [num_data, C, n_x_patches, n_y_patches, w, h]
            X_scaled = tf.transpose(
                X_scaled, [0, 2, 3, 4, 5, 1]
            )# shape [num_data, n_x_patches, n_y_patches, w, h, C]

            # here additional transposing X_scaled to shape [num_data, n_x_patches, w, n_y_patches, h, C]
            X_scaled = tf.transpose(
                X_scaled, [0, 1, 3, 2, 4, 5]
            )# shape [num_data, n_x_patches, w, n_y_patches, h, C]

            X_scaled = tf.reshape(X_scaled, input_shape)
            X_scaled = to_default_float(X_scaled)
        else:
            X_scaled = X
        return X_scaled


    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        '''
        # if X2 is None equivalent to calling with only X specified
        return self.kernel_instance.K(self.scale(X), self.scale(X2))

    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        For (most) stationary kernels scaling is not neccessary but included to keep implementation general.
        '''
        #return self.kernel_instance.K_diag(X)
        return self.kernel_instance.K_diag(self.scale(X))

    @property
    def num_patches(self) -> int:
        '''
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    self.image_shape[0] / self.patch_shape[0]
                ) * (
                    self.image_shape[1] / self.patch_shape[1]
                ) * self.colour_channels
            ),
            dtype=tf.int32
        )

class AFCPRDKernel(Kernel):
    '''
    Implements automatic full channel patch relevance detection (AFCPRD).
    Dimensions belonging to the same patch are scaled by the same lengthscale.
    This is the version presented in chapter 6.2 "Automatic Patch Relevancy Detection".
    
    Difference to APRDKernel:
    APRD scales the patches in each channel individually.
    AFCPRD uses the same lengthscale for all channels in a patch.
    
    For a plain APRD kernel an isotropic RBF kernel can be used as kernel_instance.
    Set lengthscales of kernel_instance to non-trainable to avoid parameter redundancy.
    Hyperparameter variance of kernel_instance should be trainable because AFCPRDKernel
    does not posess this parameter.
    
    For the combination of APRD with ACRD as presented in chapter 6.4 "Combination of APRD and ACRD"
    a ACRD kernel is used as kernel_instance.
    
    For combination of APRD and channel-additive kernels as briefly mentioned in
    chapter 6.5 "Channel-additive Kernels" a channel-additive kernel is used as kernel_instance.
    '''
    def __init__(
        self,
        kernel_instance: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int,
        lengthscales: Optional[TensorType] = None,
    ) -> None:
        '''
        image_shape: [2]
        patch_shape: [2]
        lengthscales: [C]
        '''
        super().__init__()
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.kernel_instance = kernel_instance
        self.colour_channels = colour_channels
        self.lengthscales = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if lengthscales is None else lengthscales,
            transform=positive()
        )

    def scale(self, X: TensorType) -> TensorType:
        '''
        X: [batch..., N, D]
        X_scaled: [batch..., N, D]
        '''
        if X is not None:
            input_shape = tf.shape(X)
            batch = tf.shape(X)[:-2]
            N = tf.shape(X)[-2]
            flat_batch = tf.reduce_prod(batch)
            num_data = flat_batch * N
            X = tf.reshape(
                X,
                [num_data, self.image_shape[0], self.image_shape[1], self.colour_channels]
            ) # shape [num_data, W, H, C]

            # if argument sizes and strides are equal it extracts disjoint patches
            Xp = tf.image.extract_patches(
                X,
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, 1, 1, 1],
                "VALID",
            ) # shape [num_data, n_x_patches, n_y_patches, w_x_h_x_C]
            shp = tf.shape(Xp)

            lengthscales_reshaped = tf.reshape(
                self.lengthscales, [1, shp[1], shp[2], 1]
            ) # shape [1, n_x_patches, n_y_patches, 1]
            lengthscales_reshaped = tf.repeat(
                lengthscales_reshaped, repeats=self.len_patches, axis=-1
            ) # shape [1, n_x_patches, n_y_patches, C_x_w_x_h]

            X_scaled = to_default_float(Xp) / to_default_float(lengthscales_reshaped)

            X_scaled = tf.reshape(
                X_scaled,
                [num_data, shp[1], shp[2], self.patch_shape[0], self.patch_shape[1], self.colour_channels]
            ) # shape [num_data, n_x_patches, n_y_patches, w, h, C]

            X_scaled = tf.transpose(
                X_scaled, [0, 1, 3, 2, 4, 5]
            )# shape [num_data, n_x_patches, w, n_y_patches, h, C]

            X_scaled = tf.reshape(X_scaled, input_shape)
            X_scaled = to_default_float(X_scaled)
        else:
            X_scaled = X
        return X_scaled


    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        '''
        # if X2 is None equivalent to calling with only X specified
        # if method scale is called with None it just returns None
        return self.kernel_instance.K(self.scale(X), self.scale(X2))

    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        For (most) stationary kernels scaling is not neccessary but included to keep implementation general.
        '''
        #return self.kernel_instance.K_diag(X)
        return self.kernel_instance.K_diag(self.scale(X))

    @property
    def num_patches(self) -> int:
        '''
        number of channels is irrelevant.
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    self.image_shape[0] / self.patch_shape[0]
                ) * (
                    self.image_shape[1] / self.patch_shape[1]
                )
            ),
            dtype=tf.int32
        )

    @property
    def len_patches(self) -> int:
        '''
        number of channels is irrelevant.
        '''
        return tf.cast(
            tf.reduce_prod(
                self.patch_shape[0] * self.patch_shape[1] * self.colour_channels
            ),
            dtype=tf.int32
        )




class AFCPRDandACRDKernel(Kernel):
    '''
    Just use APRD kernel and ACRD kernel as kernel_instance.
    '''
    pass