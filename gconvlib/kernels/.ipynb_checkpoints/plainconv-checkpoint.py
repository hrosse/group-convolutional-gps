import abc

from typing import Optional, Sequence, cast

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
from gpflow.kernels.base import Kernel

from .gconvbase import ConvolutionalBase

    
    
    
    
    
class FullConvolutional(ConvolutionalBase):
    r"""
    --- Cited from implementation of convolutional kernel in gpflow --- 
    
    Plain convolutional kernel as described in :cite:t:`vdw2017convgp`. Defines
    a GP :math:`f()` that is constructed from a sum of responses of individual patches
    in an image:

    .. math::
       f(x) = \sum_p x^{[p]}

    where :math:`x^{[p]}` is the :math:`p`'th patch in the image.

    The key reference is :cite:t:`vdw2017convgp`.
    
    --- End of citation ---

    The implementation in gpflow extracts extracts patches from each colour channel.
    This kernel is a modification of the convolutional kernel where each patch contains the values in all channels.
    Additionally the strides used in tf.image.extract_patches can be specified explicitly.
    For SVGP with inter-domain inducing variables use class InducingPatches.
    The implementation of instances k_uf and k_uu are almost identical to the ones in gpflow.
    Instead of accessing the weights directly, they are accessed via get_weights.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        weights: Optional[TensorType] = None,
        colour_channels: int = 1,
        strides: Optional[Sequence[int]] = [1, 1],
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            colour_channels=colour_channels,
            strides=strides,
            normalisation_factor_scale=normalisation_factor_scale,
        )
        self.weights = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if weights is None else weights
        )
        
        
    def get_patches(self, X: TensorType) -> tf.Tensor:
        """
        Changes compared to class Convolutional in GPflow: 
        Do not roll channel in front but hand multi-channel images as input to extraction function. 
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
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2], shp[3]]], 0)
        ) # "[batch..., N, P, S]"
        return to_default_float(reshaped_patches)
    
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        The same as implementation of plain convolutional kernel except that self.num_patches is wrapped
        in to_default_float because it returns dtype tf.int32.
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        W2 = self.weights[:, None] * self.weights[None, :] # "[P, P]"

        rank = tf.rank(Xp) - 3
        batch = tf.shape(Xp)[:-3]
        N = tf.shape(Xp)[-3]
        P = tf.shape(Xp)[-2]
        S = tf.shape(Xp)[-1]
        ones = tf.ones((rank,), dtype=tf.int32)

        if X2 is None:
            Xp = tf.reshape(Xp, tf.concat([batch, [N * P, S]], 0)) # "[batch..., N_x_P, S]"
            bigK = self.base_kernel.K(Xp) # "[batch..., N_x_P, N_x_P]"
            bigK = tf.reshape(
                bigK, tf.concat([batch, [N, P, N, P]], 0)
            ) # "[batch..., N, P, N, P]"
            W2 = tf.reshape(
                W2, tf.concat([ones, [1, P, 1, P]], 0)
            ) # "[..., 1, P, 1, P]"
            W2bigK = bigK * W2 # "[batch..., N, P, N, P]"
            return tf.reduce_sum(W2bigK, [rank + 1, rank + 3]) / (to_default_float(self.num_patches) / self.normalisation_factor_scale) ** 2.0 # "[batch..., N, N]"
        else:
            Xp2 = Xp if X2 is None else self.get_patches(X2) # "[batch2..., N2, P, S]"
            rank2 = tf.rank(Xp2) - 3
            ones2 = tf.ones((rank2,), dtype=tf.int32)
            bigK = self.base_kernel.K(Xp, Xp2) # "[batch..., N, P, batch2..., N2, P]"
            W2 = tf.reshape(
                W2, tf.concat([ones, [1, P], ones2, [1, P]], 0)
            ) #"[..., 1, P, ..., 1, P]"
            W2bigK = bigK * W2 # "[batch..., N, P, batch2..., N2, P]"
            # shape of returned final tensor is "[batch..., N, batch2..., N2]"
            return tf.reduce_sum(W2bigK, [rank + 1, rank + rank2 + 3]) / (to_default_float(self.num_patches) / self.normalisation_factor_scale) ** 2.0
    
    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        The same as implementation of plain convolutional kernel except that self.num_patches is wrapped
        in to_default_flaut because it returns dtype tf.int32.
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        rank = tf.rank(Xp) - 3
        P = tf.shape(Xp)[-2]
        ones = tf.ones((rank,), dtype=tf.int32)
        W2 = self.weights[:, None] * self.weights[None, :] # "[P, P]"
        W2 = tf.reshape(W2, tf.concat([ones, [1, P, P]], 0)) # "[..., 1, P, P]"
        bigK = self.base_kernel.K(Xp) # "[batch..., N, P, P]"
        return tf.reduce_sum(bigK * W2, [rank + 1, rank + 2]) / (to_default_float(self.num_patches) / self.normalisation_factor_scale) ** 2.0
    
    
    @property
    def get_weights(self) -> TensorType:
        '''
        '''
        return self.weights
    
    
    @property
    def num_patches(self) -> int:
        '''
        Change: 
        Removed factor colour_channels.
        Use correct equation to compute number of patches because strides could be greater than one.
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    tf.floor((self.image_shape[0] - self.patch_shape[0]) / self.strides[0]) + 1
                ) * (
                    tf.floor((self.image_shape[1] - self.patch_shape[1]) / self.strides[1]) + 1
                )
            ),
            dtype=tf.int32
        )



# not used
class FullConvolutionalLoopX(ConvolutionalBase):
    r"""
    This is just the implementation of an idea but not used and not tested.
    
    This kernel is a modification of the convolutional kernel where each patch contains all channels.
    Additionally the strides used in tf.image.extract_patches can be specified explicitly.
    Method K_diag is implemented using a Tensorflow while loop to allow more control over parallelisation.
    
    For SVGP with inter-domain inducing variables use class InducingPatches or InducingPatchesLoopX.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        weights: Optional[TensorType] = None,
        colour_channels: int = 1,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 5,
        max_parallel_iterations_kuf: int = 10,
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            colour_channels=colour_channels,
            strides=strides,
        )
        self.weights = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if weights is None else weights
        )
        self.max_parallel_iterations_kdiag = max_parallel_iterations_kdiag
        self.max_parallel_iterations_kuf = max_parallel_iterations_kuf
        self.swap_memory = True
        
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
        reshaped_patches = tf.reshape(
            patches, 
            tf.concat([batch, [N, shp[1] * shp[2], shp[3]]], 0)
        ) # "[batch..., N, P, S]"
        return to_default_float(reshaped_patches)
    
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        The same as implementation of plain convolutional kernel except that self.num_patches is wrapped
        in to_default_flaut because it returns dtype tf.int32.
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        W2 = self.weights[:, None] * self.weights[None, :] # "[P, P]"

        rank = tf.rank(Xp) - 3
        batch = tf.shape(Xp)[:-3]
        N = tf.shape(Xp)[-3]
        P = tf.shape(Xp)[-2]
        S = tf.shape(Xp)[-1]
        ones = tf.ones((rank,), dtype=tf.int32)

        if X2 is None:
            Xp = tf.reshape(Xp, tf.concat([batch, [N * P, S]], 0)) # "[batch..., N_x_P, S]"
            bigK = self.base_kernel.K(Xp) # "[batch..., N_x_P, N_x_P]"
            bigK = tf.reshape(
                bigK, tf.concat([batch, [N, P, N, P]], 0)
            ) # "[batch..., N, P, N, P]"
            W2 = tf.reshape(
                W2, tf.concat([ones, [1, P, 1, P]], 0)
            ) # "[..., 1, P, 1, P]"
            W2bigK = bigK * W2 # "[batch..., N, P, N, P]"
            return tf.reduce_sum(W2bigK, [rank + 1, rank + 3]) / to_default_float(self.num_patches) ** 2.0 # "[batch..., N, N]"
        else:
            Xp2 = Xp if X2 is None else self.get_patches(X2) # "[batch2..., N2, P, S]"
            rank2 = tf.rank(Xp2) - 3
            ones2 = tf.ones((rank2,), dtype=tf.int32)
            bigK = self.base_kernel.K(Xp, Xp2) # "[batch..., N, P, batch2..., N2, P]"
            W2 = tf.reshape(
                W2, tf.concat([ones, [1, P], ones2, [1, P]], 0)
            ) #"[..., 1, P, ..., 1, P]"
            W2bigK = bigK * W2 # "[batch..., N, P, batch2..., N2, P]"
            # shape of returned final tensor is "[batch..., N, batch2..., N2]"
            return tf.reduce_sum(W2bigK, [rank + 1, rank + rank2 + 3]) / to_default_float(self.num_patches) ** 2.0
    
    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        Loop is used to compute entries individually.
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        rank = tf.rank(Xp) - 3 
        batch = tf.shape(Xp)[:-3]
        N = tf.shape(Xp)[-3]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        P = tf.shape(Xp)[-2]
        S = tf.shape(Xp)[-1]
        Xp = tf.reshape(Xp, [num_data, P, S])
        W2 = self.get_weights[:, None] * self.get_weights[None, :] # "[P, P]"
        
        i0 = tf.constant(0)
        kdiag_container = tf.TensorArray(default_float(), size=num_data, clear_after_read=True)
        
        def loop_cond(i, _):
            return tf.less(i, num_data)
        
        def loop_body(j, kdiag_TensorArray):
            kdiag_entry = self.base_kernel(Xp[j], Xp[j]) # shape [P, P]
            kdiag_entry = tf.reduce_sum(W2 * kdiag_entry) 
            kdiag_TensorArray = kdiag_TensorArray.write(j, kdiag_entry)
            #kdiag_TensorArray.mark_used()
            return [j+1, kdiag_TensorArray]
        
        result = tf.while_loop(
            loop_cond, loop_body, [i0, kdiag_container],
            parallel_iterations=self.max_parallel_iterations_kdiag,
            swap_memory=self.swap_memory
        )
        kdiag_evals = result[1].stack() # shape (num_data,)
        kdiag_evals = tf.reshape(
            kdiag_evals,
            tf.concat([batch, [N]], 0)
        ) # shape [batch..., N]
        norm_const = to_default_float(self.num_patches) ** 2.0 # P^2
        return kdiag_evals / norm_const
    
    def Kuf(self, Z: TensorType, X: TensorType) -> tf.Tensor:
        '''
        '''
        Xp = self.get_patches(X)  # [N, num_patches, patch_len]
        N = tf.shape(Xp)[0]
        Z = to_default_float(Z)
        weights = to_default_float(self.get_weights)
    
        i0 = tf.constant(0)
        kuf_container = tf.TensorArray(default_float(), size=N, clear_after_read=True)
            
        def loop_cond(i, _):
            return tf.less(i, N)
            
        def loop_body(j, kuf_TensorArray):
            kuf_entry = self.base_kernel(Z, Xp[j]) # shape [M, P]
            kuf_entry = tf.reduce_sum(kuf_entry * weights, [1]) # shape (M)
            return (j+1, kuf_TensorArray.write(j, kuf_entry))
            
        result = tf.while_loop(
        loop_cond, loop_body, (i0, kuf_container),
            parallel_iterations=self.max_parallel_iterations_kuf, swap_memory=self.swap_memory
        )
        kuf_evals = result[1].stack() # shape (N, M)
        kuf_evals = tf.transpose(kuf_evals, [1,0]) # shape (M, N)
        return kuf_evals / to_default_float(self.num_patches)
    
    @property
    def get_weights(self) -> TensorType:
        return self.weights
    
    
    @property
    def num_patches(self) -> int:
        '''
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    tf.floor((self.image_shape[0] - self.patch_shape[0]) / self.strides[0]) + 1
                ) * (
                    tf.floor((self.image_shape[1] - self.patch_shape[1]) / self.strides[1]) + 1
                )
            ),
            dtype=tf.int32
        )