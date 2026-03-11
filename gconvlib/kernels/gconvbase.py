import abc

from typing import Optional, Sequence, cast

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
from gpflow.kernels.base import Kernel

# Contains the base classes for all types of convolutional kernels.

class ConvolutionalBase(Kernel):
    r"""
    All 
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        colour_channels: int = 1,
        strides: Optional[Sequence[int]] = [1, 1],
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__()
        self.base_kernel = base_kernel
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.colour_channels = colour_channels
        self.strides = strides
        self.normalisation_factor_scale = to_default_float(normalisation_factor_scale)
    
    @abc.abstractmethod
    def get_patches(self, X: TensorType) -> tf.Tensor:
        raise NotImplementedError
        
    @property
    @abc.abstractmethod
    def get_weights(self) -> TensorType:
        '''
        General and convenient method to access weights in the right shape and size for further use.
        E.g. for multiplicating with tensor containing base kernel evaluations.
        Ordering of weights, i.e. association with corresponding patches is assured by following some convention in method get_patches.
        '''
        raise NotImplementedError
        
    @property
    @abc.abstractmethod
    def num_patches(self) -> int:
        '''
        Number of patches depends on image shape, patch shape, group order and strides.
        '''
        raise NotImplementedError
        
        
        


class GConvolutionalBase(ConvolutionalBase):
    r"""
    Base class for the group-convolutional kernels with full weights as well as with weights of form w_pg = w_p * w_g.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        grouporder: int,
        colour_channels: int = 1,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 1000,
        max_parallel_iterations_kuf: int = 1000,
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
        self.grouporder = grouporder
        self.max_parallel_iterations_kdiag = max_parallel_iterations_kdiag
        self.max_parallel_iterations_kuf = max_parallel_iterations_kuf
        self.swap_memory = False
        
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        The same as implementation of plain convolutional kernel.
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        W2 = self.get_weights[:, None] * self.get_weights[None, :] # "[P, P]"

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
            return tf.reduce_sum(W2bigK, [rank + 1, rank + 3]) * (self.normalisation_factor_scale/ to_default_float(self.num_patches)) ** 2.0 # "[batch..., N, N]"

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
            return tf.reduce_sum(W2bigK, [rank + 1, rank + rank2 + 3]) * (self.normalisation_factor_scale/ to_default_float(self.num_patches)) ** 2.0
        
    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        rank = tf.rank(Xp) - 3
        P = tf.shape(Xp)[-2]
        ones = tf.ones((rank,), dtype=tf.int32)
        W2 = self.get_weights[:, None] * self.get_weights[None, :] # "[P, P]"
        W2 = tf.reshape(W2, tf.concat([ones, [1, P, P]], 0)) # "[..., 1, P, P]"
        bigK = self.base_kernel.K(Xp) # "[batch..., N, P, P]"
        return tf.reduce_sum(bigK * W2, [rank + 1, rank + 2]) / (to_default_float(self.num_patches) / self.normalisation_factor_scale) ** 2.0
    
    @property
    def num_patches(self) -> int:
        '''
        Number of transformed patches.
        Increases by factor group order.
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    tf.floor((self.image_shape[0] - self.patch_shape[0]) / self.strides[0]) + 1
                ) * (
                    tf.floor((self.image_shape[1] - self.patch_shape[1]) / self.strides[1]) + 1
                ) * self.grouporder
            ),
            dtype=tf.int32
        )
    
    
    
    
# not used
class GConvolutionalBaseLoopDiag(ConvolutionalBase):
    r"""
    This class has not yet been tested.
    This is just an implementation for some rough idea.
    
    The initial idea is to iterate over elments in a minibatch using a tensorflow while_loop (and/or over 
    inducing patches when evaluating k_fu) and set argument swap_memory to True so that tensors needed for 
    backpropagation during optimisation are stored in CPU memory.
    The values of max_parallel_iterations_kuf and max_parallel_iterations_kdiag could be used as argument
    parallel_iterations in tf.while_loop for control over memory consumption and parallelisation.

    Moving data between GPU and CPU memory creates overhead.
    But even if sufficient computational ressources were available to run large and costly versions of 
    group-convolutional GP models training for a long time, memory consumption would prohibit utilisation 
    of GPUs for training.

    Example scenario for multi-class:
        - For each class train a binary classification model (one-vs-rest)
        - Initialise multi-class model with results from pre-training
        - Use the approach here to train the full model
        - For efficient inference set swap_memory to False
        - Alternatively: Initialise full model with result of training
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        grouporder: int,
        colour_channels: int = 1,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 5,
        max_parallel_iterations_kuf: int = 10,
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
        self.grouporder = grouporder
        self.max_parallel_iterations_kdiag = max_parallel_iterations_kdiag
        self.max_parallel_iterations_kuf = max_parallel_iterations_kuf
        # Memory swapping in combination with InducingPatchesLoopX can be memory-efficient 
        # but propably very expensive during training.
        self.swap_memory = True
        
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        W2 = self.get_weights[:, None] * self.get_weights[None, :] # "[P, P]"

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
        A total of ord(G)^2 * P^2 base kernel are evaluated in the body of the loop.
        '''
        # Here P = num_patches_image * ord(G)
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
        Here the kernel matrix Kuf is computed by iterating over the N points in X and evaluating 
        the interdomain instance k_uf of the kernel individually.
        In principle, it is also possible to loop over the inducing patches Z.
        When looping over the inputs X a total of ord(G) * P * M base kernels are evaluated
        in the body of the loop.
        When looping over the inducing patches Z a total of N * ord(G) * P base kernels are evaluated
        in the body of the loop.
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
    def num_patches(self) -> int:
        '''
        Number of transformed patches.
        Increases by factor group order
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    tf.floor((self.image_shape[0] - self.patch_shape[0]) / self.strides[0]) + 1
                ) * (
                    tf.floor((self.image_shape[1] - self.patch_shape[1]) / self.strides[1]) + 1
                ) * self.grouporder
            ),
            dtype=tf.int32
        )
    
    
    
    
    
class GConvolutional(GConvolutionalBase):
    r"""
    Base class for the group-convolutional kernels where weights are defined as w_pg = w_p * w_g.
    So far no experiments have been done with this kind of group-convolutional kernel.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        grouporder: int,
        colour_channels: int = 1,
        patch_weights: Optional[TensorType] = None,
        groupelement_weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 250,
        max_parallel_iterations_kuf: int = 250,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=grouporder,
            colour_channels=colour_channels,
            strides=strides,
            max_parallel_iterations_kdiag=max_parallel_iterations_kdiag,
            max_parallel_iterations_kuf=max_parallel_iterations_kuf,
            normalisation_factor_scale=normalisation_factor_scale,
        )
        self.patch_weights = Parameter(
            np.ones(self.num_patches_image, dtype=default_float()) if patch_weights is None else patch_weights
        )
        self.groupelement_weights = Parameter(
            np.ones(self.grouporder, dtype=default_float()) if groupelement_weights is None else groupelement_weights
        )
        
    @property
    def get_weights(self) -> TensorType:
        weights = self.patch_weights[:, None] * self.groupelement_weights[None, :] # "[P, ord(G)]"
        weights = tf.reshape(weights, [self.num_patches_image * self.grouporder])
        return weights
    
    @property
    def num_patches_image(self) -> int:
        '''
        Number of patches in image.
        Required in method get_weights.
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
    
    
    
    
    
class GConvolutionalFullWeights(GConvolutionalBase):
    r"""
    Base class for the group-convolutional kernels with full weights.
    This is the version where each group element at a specific patch location has its own weight.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        grouporder: int,
        colour_channels: int = 1,
        weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 250,
        max_parallel_iterations_kuf: int = 250,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=grouporder,
            colour_channels=colour_channels,
            strides=strides,
            max_parallel_iterations_kdiag=max_parallel_iterations_kdiag,
            max_parallel_iterations_kuf=max_parallel_iterations_kuf,
            normalisation_factor_scale=normalisation_factor_scale,
        )
        self.weights = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if weights is None else weights
        )
        
    @property
    def get_weights(self) -> TensorType:
        return self.weights
    
    
    
    
    
class GInvarConvolutionalBase(ConvolutionalBase):
    r"""
    Base class for the group-convolutional kernels with weights where w_pg = w_p.
    This includes the group-invariant convolutional kernels with weights that are constrained
    to the fixed points under the action of a group (i.e. the group-invariant convolutional GPs).

    In both cases K_diag can be computed with a single sum over the group elements as described
    in chapter 5.2.3 "Interdomain Inducing Variables for Group-Convolutional Gaussian Processes".
    In principle one of the sums could be removed in method K, too.
    Although in most cases GPs with group-convolutional kernels are infeasible without interdomain sparse approximation.

    In K_diag the entries are computed individually using a tensorflow while_loop.
    Attributes max_parallel_iterations_kdiag and swap_memory are given to tf.while_loop to make sure
    that parallelisation on GPU is utilised and there is no overhead from reading and writing to memory.
    This might not be the most efficient approach but significant improvements in terms of memory consumption 
    and computational cost can be observed.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        grouporder: int,
        colour_channels: int = 1,
        weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 250,
        max_parallel_iterations_kuf: int = 250,
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
        self.grouporder = grouporder
        self.weights = Parameter(
            np.ones(self.num_patches_image, dtype=default_float()) if weights is None else weights
        )
        self.max_parallel_iterations_kdiag = max_parallel_iterations_kdiag
        self.max_parallel_iterations_kuf = max_parallel_iterations_kuf
        self.swap_memory = False
        
    def get_patches_image(self, X: TensorType) -> tf.Tensor:
        """
        Extract patches from image as would be done in a standard convolutional kernel.
        Required in K_diag.
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
        TODO: Change so that double-sum is removed.
        '''
        Xp = self.get_patches(X) # "[batch..., N, P, S]"
        W2 = self.get_weights[:, None] * self.get_weights[None, :] # "[P, P]"

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
            return tf.reduce_sum(W2bigK, [rank + 1, rank + 3]) * (self.normalisation_factor_scale / to_default_float(self.num_patches)) ** 2.0 # "[batch..., N, N]"

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
            return tf.reduce_sum(W2bigK, [rank + 1, rank + rank2 + 3]) * (self.normalisation_factor_scale / to_default_float(self.num_patches)) ** 2.0
    
    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        '''
        Xpg = self.get_patches(X) # "[batch..., N, P*ord(G), S]"
        Xp = self.get_patches_image(X) # "[batch..., N, P, S]"
        rank = tf.rank(Xp) - 3 # Xpg and Xp have same rank.
        batch = tf.shape(Xp)[:-3]
        N = tf.shape(Xp)[-3]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        P = tf.shape(Xp)[-2]
        PG = tf.shape(Xpg)[-2]
        S = tf.shape(Xp)[-1]
        Xp = tf.reshape(Xp, [num_data, P, S])
        Xpg = tf.reshape(Xpg, [num_data, PG, S])
        W2 = self.get_weights[:, None] * self.weights[None, :] # "[P*ord(G), P]"
        
        i0 = tf.constant(0)
        kdiag_container = tf.TensorArray(default_float(), size=num_data, clear_after_read=True)
        
        def loop_cond(i, _):
            return tf.less(i, num_data)
        
        def loop_body(j, kdiag_TensorArray):
            kdiag_entry = self.base_kernel(Xpg[j], Xp[j]) # shape [P*ord(G), P]
            kdiag_entry = tf.reduce_sum(W2 * kdiag_entry) # just a scalar
            kdiag_TensorArray = kdiag_TensorArray.write(j, kdiag_entry)
            return (j+1, kdiag_TensorArray)
        
        result = tf.while_loop(
            loop_cond, loop_body, (i0, kdiag_container),
            parallel_iterations=self.max_parallel_iterations_kdiag,
            swap_memory=self.swap_memory
        )
        kdiag_evals = result[1].stack() # shape (num_data,)
        kdiag_evals = tf.reshape(
            kdiag_evals,
            tf.concat([batch, [N]], 0)
        ) # shape [batch..., N]
        norm_const = to_default_float(self.grouporder) * to_default_float(self.num_patches_image) ** 2.0 # |G|*P^2
        return kdiag_evals / (norm_const / self.normalisation_factor_scale ** 2)
        
    @property
    def num_patches(self) -> int:
        '''
        Number of transformed patches.
        Increases by factor group order.
        '''
        return tf.cast(
            tf.reduce_prod(
                (
                    tf.floor((self.image_shape[0] - self.patch_shape[0]) / self.strides[0]) + 1
                ) * (
                    tf.floor((self.image_shape[1] - self.patch_shape[1]) / self.strides[1]) + 1
                ) * self.grouporder
            ),
            dtype=tf.int32
        )
    
    @property
    def num_patches_image(self) -> int:
        '''
        Number of patches in image.
        Required in method get_weights and K_diag.
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
    
    @property
    def num_patches_image(self) -> int:
        '''
        Number of patches in image.
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
    
    
    
    
    
class GInvarPatchesConvolutional(GInvarConvolutionalBase):
    r"""
    Base class for the group-convolutional kernels with weights where w_pg = w_p without any further
    constraints on the patch location weights.
    Effectively a convolutional GP with invariant patch-response function.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        grouporder: int,
        colour_channels: int = 1,
        weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 250,
        max_parallel_iterations_kuf: int = 250,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=grouporder,
            colour_channels=colour_channels,
            weights=weights,
            strides=strides,
            max_parallel_iterations_kdiag=max_parallel_iterations_kdiag,
            max_parallel_iterations_kuf=max_parallel_iterations_kuf,
            normalisation_factor_scale=normalisation_factor_scale,
        )
        
    @property
    def get_weights(self) -> TensorType:
        # Below dtype is specified as default_float from gpflow.config to ensure matching data types.
        ones = tf.ones((self.grouporder,), dtype=default_float())
        weights = self.weights[:, None] * ones[None, :] # "[P, ord(G)]"
        weights = tf.reshape(weights, [self.num_patches_image * self.grouporder])
        return weights
    
    
    
    
    
class GInvarConvolutional(GInvarConvolutionalBase):
    r"""
    Base class for the group-invariant convolutional kernels where the patch weights 
    are constrained by mapping them into the space of fixed points under the action of a group.
    
    The map into the space of fixed points is implemented in the inheriting kernel classes.
    The background for this kind of weight-sharing is described in chapter 5.1.4 "Equivariant Weighted Kernel Maps", 
    chapter 5.1.5 "Comparison of Permutation-invariant and Convolutional Kernel" 
    and chapter 5.2.2 "Weighted Group-Convolutional Kernel".
    A notebook in the supplement illustrates the concept on a simple example.
    The weights are treated as if they were a single-channel image.
    Methods W_weights and H_weights are used to reshape the weights accordingly.
    An additional illustration can be found in chapter 5.3 "Comparison with Kernel Networks" where the mean function of
    the approximate posterior is depicted as a kernel network.
    The evaluation of the interdomain kernel instance k_fu for one single inducing patch is interpreted as the sum 
    over a W x H x order(G) feature map.
    The implementation corresponds to first summing over the last axis of each feature map and then multiplying with
    the weights.
    """
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        grouporder: int,
        colour_channels: int = 1,
        weights: Optional[TensorType] = None,
        strides: Optional[Sequence[int]] = [1, 1],
        max_parallel_iterations_kdiag: int = 250,
        max_parallel_iterations_kuf: int = 250,
        normalisation_factor_scale: int = 1
    ) -> None:
        '''
        '''
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            patch_shape=patch_shape,
            grouporder=grouporder,
            colour_channels=colour_channels,
            weights=weights,
            strides=strides,
            max_parallel_iterations_kdiag=max_parallel_iterations_kdiag,
            max_parallel_iterations_kuf=max_parallel_iterations_kuf,
            normalisation_factor_scale=normalisation_factor_scale,
        )
    
    @property
    def W_weights(self) -> TensorType:
        '''
        '''
        return tf.cast(
            tf.floor((self.image_shape[0] - self.patch_shape[0]) / self.strides[0]) + 1, 
            dtype=tf.int32
        )
    
    @property
    def H_weights(self) -> TensorType:
        '''
        '''
        return tf.cast(
            tf.floor((self.image_shape[1] - self.patch_shape[1]) / self.strides[1]) + 1, 
            dtype=tf.int32
        )