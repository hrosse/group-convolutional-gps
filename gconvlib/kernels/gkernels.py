import abc

from typing import Optional, Sequence, cast

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorType
from gpflow.config import default_float
from gpflow.utilities import to_default_float, positive
from gpflow.kernels.base import Kernel


### Contains base class and different versions of the GKernel ###



# All versions of the G-kernel inherit from the class
class GKernelBase(Kernel):
    '''
    Base class for all kernel that are symmetrised with respect to some group action on the input images.
    Classes derived from this class require implementation of specific get_X_transformed and specify group order.
    Set weights to one and set weights to non-trainable for an argument-wise invariant kernel.
    That is the kernel obtained via "average-over-orbit" method.
    Implementation is similar to the kernel class "Convolutional" in GPflow.

    Kernel classes inheriting from this class are the G-Kernel described in chapter 5.1.3 "Weighted Kernel Maps".
    '''
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        n_channel: int,
        grouporder: int,
        weights: Optional[TensorType],
    )-> None:
        '''        
        input:
        
        '''
        super().__init__()
        self.image_shape = image_shape # width, height
        self.W = image_shape[0]
        self.H = image_shape[1]
        self.n_channel = n_channel # number of channels
        self.grouporder = grouporder
        self.base_kernel = base_kernel
        self.weights = Parameter(
            np.ones(self.grouporder, dtype=default_float()) if weights is None else weights
        )
    
    
    @abc.abstractmethod
    def get_X_transformed(self, X: TensorType) -> tf.Tensor:
        '''
        input:
        X: [batch..., N, D]
    
        return:
        X_transformed: [batch..., N, ord(G), D]
        '''
        raise NotImplementedError
      
    
    
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        '''
        '''
        X = self.get_X_transformed(X) # [batch..., N, ord(G), D]
        Wg = self.weights[:, None] * self.weights[None, :] # [ord(G), ord(G)]
        rank = tf.rank(X) - 3 # Rank of [batch..., N, ord(G), D], excluding last 3 axis
        batch = tf.shape(X)[:-3]
        N = tf.shape(X)[-3]
        ordG = tf.shape(X)[-2]
        D = tf.shape(X)[-1]
        ones = tf.ones((rank,), dtype=tf.int32)
        if X2 is None: # For input of type K(X)
            X = tf.reshape(X, tf.concat([batch, [N*ordG, D]], 0)) # [batch..., N*ord(G), D]
            K_mat = self.base_kernel.K(X) # [batch..., N*ord(G), N*ord(G)] 
            K_mat = tf.reshape(K_mat, tf.concat([batch, [N, ordG, N, ordG]], 0)) # "[batch..., N, ord(G), N, ord(G)]"
            Wg = tf.reshape(Wg, tf.concat([ones, [1, ordG, 1, ordG]], 0)) #"[..., 1, ord(G), 1, ord(G)]"
            return tf.reduce_sum(K_mat * Wg, [rank + 1, rank + 3]) / self.grouporder**2.0 #"[batch..., N, N]"
        else: # For input of type K(X,X_prime)
            X2 = X if X2 is None else self.get_X_transformed(X2) # [batch2..., N2, ord(G), D]
            rank2 = tf.rank(X2) - 3
            ones2 = tf.ones((rank2,), dtype=tf.int32)
            K_mat = self.base_kernel.K(X, X2) # [batch..., N, ord(G), batch2..., N2, ord(G)]
            Wg = tf.reshape(Wg, tf.concat([ones, [1, ordG], ones2, [1, ordG]], 0)) #"[..., 1, ord(G), ..., 1, ord(G)]"
            return tf.reduce_sum(K_mat * Wg, [rank + 1, rank+rank2 + 3]) / self.grouporder**2.0 # [batch..., N, batch2..., N2]
        
    def K_diag(self, X: TensorType) -> tf.Tensor:
        '''
        '''
        X = self.get_X_transformed(X) # [batch..., N, ord(G), D]
        rank = tf.rank(X) - 3 # Rank of [batch..., N, ord(G), D], excluding last 3 axis
        ordG = tf.shape(X)[-2]
        ones = tf.ones((rank,), dtype=tf.int32)
        Wg = self.weights[:, None] * self.weights[None, :] # [ord(G), ord(G)]
        Wg = tf.reshape(Wg, tf.concat([ones, [1, ordG, ordG]], 0)) # "[..., 1, ord(G), ord(G)]"
        K_mat = self.base_kernel.K(X) # [batch..., N, ord(G), ord(G)]
        return tf.reduce_sum(K_mat * Wg, [rank + 1, rank + 2]) / self.grouporder**2.0 # [batch..., N]
    
    
    


### Implementations of G-Kernel for specific transformations (rotations, reflections and compositions)

# Since the number of weights is relatively small, it is possible to visually inspect them.
# Each weight corresponds to a specific transformation so that the values tell us something about
# the symmetries in the data that have been discovered.
#
# Example MNIST: 
# When testing a G-Kernel where the group action corresponds to image rotations by 180 degree
# on 6-vs-9 from original MNIST, we achieved near-perfect classification on the test data.
# Only one point was on the wrong side of the decision boundary (one half).
# For the misclassified point the prediction of belonging to class one had a probability somewhere close to 0.6.
# The weight for identity was a positive value and the weight for rotation by 180 degree was a negative value.
# As would be expected.
#
# Example CIFAR-10:
# D4-Kernel had relatively large positive weights for identity and for left-right flip across the classes.
# On closer inspection, this symmetry can visually be seen in the images.

class D4Kernel(GKernelBase):
    '''
    '''
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        n_channel: int = 1,
        weights: Optional[TensorType] = None,
    )-> None:
        '''        
        Group order is hardcoded here.
        '''
        grouporder = 8 
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            n_channel=n_channel,
            grouporder=grouporder,
            weights=weights,
        )
    
    def get_X_transformed(self, X: TensorType) -> tf.Tensor:
        '''
        input:
        X: [batch..., N, D]
    
        return:
        X_transformed: [batch..., N, ord(G), D]
        '''
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        D = tf.shape(X)[-1]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(X, [num_data, self.W, self.H, self.n_channel])
        # Naming convention: Xn stands for the rotation degree, Xf or Xnf for degree and then flip.
        # Ordering convention: 0, 90, 180, 270 counter-clockwise, then flip
        X90 = tf.transpose(tf.reverse(X, [-2]), [0, 2, 1, 3])
        X180 = tf.reverse(X, axis=[-2, -3])
        X270 = tf.reverse(tf.transpose(X, [0, 2, 1, 3]), [-2])
        Xf = tf.reverse(X, axis=[-2])
        X90f = tf.reverse(X90, axis=[-2])
        X180f = tf.reverse(X180, axis=[-2])
        X270f = tf.reverse(X270, axis=[-2])
        X = tf.stack(
            [X, X90, X180, X270, Xf, X90f, X180f, X270f], axis=1
        ) # [num_data, ord(G), W, H, self.n_channel]
        final_shape = tf.concat([batch, [N, self.grouporder, D]], 0)
        return to_default_float(tf.reshape(X, final_shape)) # [batch..., N, ord(G), D]
    
    
    
    
    
class D2Kernel(GKernelBase):
    '''
    '''
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        n_channel: int = 1,
        weights: Optional[TensorType] = None,
    )-> None:
        '''        
        Group order is hardcoded here.
        '''
        grouporder = 4 
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            n_channel=n_channel,
            grouporder=grouporder,
            weights=weights,
        )
        
    def get_X_transformed(self, X: TensorType) -> tf.Tensor:
        '''
        input:
        X: [batch..., N, D]
    
        return:
        X_transformed: [batch..., N, ord(G), D]
        '''
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        D = tf.shape(X)[-1]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(X, [num_data, self.W, self.H, self.n_channel])
        # Naming convention: h for horizontal and v for vertical flip.
        # Ordering convention: first horizontal flip than vertical flips.
        Xh = tf.reverse(X, axis=[-2])
        Xv = tf.reverse(X, axis=[-3])
        Xhv = tf.reverse(X, axis=[-2, -3])
        X = tf.stack([X, Xh, Xv, Xhv], axis=1) # [num_data, ord(G), W, H, self.n_channel]
        final_shape = tf.concat([batch, [N, self.grouporder, D]], 0)
        return to_default_float(tf.reshape(X, final_shape)) # [batch..., N, ord(G), D]
    
    
    
    
    
class RotationKernel(GKernelBase):
    '''
    '''
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        n_channel: int = 1,
        weights: Optional[TensorType] = None,
    )-> None:
        '''        
        Group order is hardcoded here.
        '''
        grouporder = 4 
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            n_channel=n_channel,
            grouporder=grouporder,
            weights=weights,
        )
        
    def get_X_transformed(self, X: TensorType) -> tf.Tensor:
        '''
        input:
        X: [batch..., N, D]
    
        return:
        X_transformed: [batch..., N, ord(G), D]
        '''
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        D = tf.shape(X)[-1]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(X, [num_data, self.W, self.H, self.n_channel])
        # Naming convention: Xn stands for the rotation degree.
        # Ordering convention: 0, 90, 180, 270 counter-clockwise
        X90 = tf.transpose(tf.reverse(X, [-2]), [0, 2, 1, 3])
        X180 = tf.reverse(X, axis=[-2, -3])
        X270 = tf.reverse(tf.transpose(X, [0, 2, 1, 3]), [-1])
        X = tf.stack([X, X90, X180, X270], axis=1) # [num_data, ord(G), W, H, self.n_channel]
        final_shape = tf.concat([batch, [N, self.grouporder, D]], 0)
        return to_default_float(tf.reshape(X, final_shape)) # [batch..., N, ord(G), D]
    
    
    
    
    
class Rotation180Kernel(GKernelBase):
    '''
    '''
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        n_channel: int = 1,
        weights: Optional[TensorType] = None,
    )-> None:
        '''        
        Group order is hardcoded here.
        '''
        grouporder = 2 
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            n_channel=n_channel,
            grouporder=grouporder,
            weights=weights,
        )
        
    def get_X_transformed(self, X: TensorType) -> tf.Tensor:
        '''
        input:
        X: [batch..., N, D]
    
        return:
        X_transformed: [batch..., N, ord(G), D]
        '''
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        D = tf.shape(X)[-1]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(X, [num_data, self.W, self.H, self.n_channel])
        Xr = tf.reverse(X, axis=[-2, -3])
        X = tf.stack([X, Xr], axis=1) # [num_data, ord(G), W, H, self.n_channel]
        final_shape = tf.concat([batch, [N, self.grouporder, D]], 0)
        return to_default_float(tf.reshape(X, final_shape)) # [batch..., N, ord(G), D]
    
    
    
    
    
class FlipLeftRightKernel(GKernelBase):
    '''
    '''
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        n_channel: int = 1,
        weights: Optional[TensorType] = None,
    )-> None:
        '''        
        Group order is hardcoded here.
        '''
        grouporder = 2 
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            n_channel=n_channel,
            grouporder=grouporder,
            weights=weights,
        )
        
    def get_X_transformed(self, X: TensorType) -> tf.Tensor:
        '''
        input:
        X: [batch..., N, D]
    
        return:
        X_transformed: [batch..., N, ord(G), D]
        '''
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        D = tf.shape(X)[-1]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(X, [num_data, self.W, self.H, self.n_channel])
        Xf = tf.reverse(X, axis=[-2])
        X = tf.stack([X, Xf], axis=1) # [num_data, ord(G), W, H, self.n_channel]
        final_shape = tf.concat([batch, [N, self.grouporder, D]], 0)
        return to_default_float(tf.reshape(X, final_shape)) # [batch..., N, ord(G), D]
    
    
    
    
    
class FlipUpDownKernel(GKernelBase):
    '''
    '''
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        n_channel: int = 1,
        weights: Optional[TensorType] = None,
    )-> None:
        '''        
        Group order is hardcoded here.
        '''
        grouporder = 2 
        super().__init__(
            base_kernel=base_kernel,
            image_shape=image_shape,
            n_channel=n_channel,
            grouporder=grouporder,
            weights=weights,
        )
        
    def get_X_transformed(self, X: TensorType) -> tf.Tensor:
        '''
        input:
        X: [batch..., N, D]
    
        return:
        X_transformed: [batch..., N, ord(G), D]
        '''
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        D = tf.shape(X)[-1]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = tf.reshape(X, [num_data, self.W, self.H, self.n_channel])
        Xf = tf.reverse(X, axis=[-3])
        X = tf.stack([X, Xf], axis=1) # [num_data, ord(G), W, H, self.n_channel]
        final_shape = tf.concat([batch, [N, self.grouporder, D]], 0)
        return to_default_float(tf.reshape(X, final_shape)) # [batch..., N, ord(G), D]