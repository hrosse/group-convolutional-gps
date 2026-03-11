import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Callable, Optional, Type

import gpflow

from gpflow.base import TensorType, MeanAndVariance
from gpflow import logdensities

from gpflow.likelihoods.multiclass import Softmax
from gpflow.likelihoods.multilatent import MultiLatentLikelihood
from gpflow.likelihoods.utils import inv_probit



class SoftmaxFixed(Softmax):
    '''
    Modification to method _log_prob required.
    See: https://github.com/GPflow/GPflow/issues/1591
    '''
    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=tf.argmax(Y, axis=1))
    
    


### not used
#
# For binary classification with multiple latent GPs and inducing points in different spaces.
# Could be used for experiments such as one-vs-all with CIFAR-10.
# For example to investigate behaviour of different models in more detail.
# Or for pretraining and models selection per class.
class MultilatentBernoulli(MultiLatentLikelihood):
    '''
    Mean field approximation given two or more a priori independent latent GPs.
    '''
    def __init__(self, 
                 latent_dim: int,
                 invlink: Callable[[tf.Tensor], tf.Tensor] = inv_probit, 
                 **kwargs: Any
                ) -> None:
        
        super().__init__(
            latent_dim=latent_dim,
            **kwargs,
        )
        self.invlink = invlink
        
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        '''
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :returns: invlink(mean) [..., 1]
        '''
        Fsum = tf.expand_dims(tf.reduce_sum(F, axis=-1),1) # Sum means of latent GP posteriors
        return self.invlink(Fsum)
    
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        p = self.conditional_mean(X, F)
        return p - (p ** 2)
    
    def _predict_mean_and_var(
        self, 
        X: TensorType, 
        Fmu: TensorType, 
        Fvar: TensorType
    ) -> MeanAndVariance:
        '''
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        '''
        if self.invlink is inv_probit:
            FmuSum = tf.expand_dims(tf.reduce_sum(Fmu, axis=-1),1) # Sum means of latent GP posteriors
            FvarSum = tf.expand_dims(tf.reduce_sum(Fvar, axis=-1),1) # Sum variance of latent GP posteriors (only diags)
            p = self.invlink(FmuSum / tf.sqrt(1 + FvarSum))
            return p, p - tf.square(p)
        else:
            return super()._predict_mean_and_var(X, Fmu, Fvar)
        
    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        """
        The log probability density log p(Y|F)

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., 1]:
        :returns: log density Bernoulli, with shape [...]
        """
        Fsum = tf.reduce_sum(F, axis=-1, keepdims=True) # Sum means of latent GP posteriors
        return tf.reduce_sum(logdensities.bernoulli(Y, self.invlink(Fsum)), axis=-1)
    
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        p = self.predict_mean_and_var(X, Fmu, Fvar)[0]
        return tf.reduce_sum(logdensities.bernoulli(Y, p), axis=-1)
    
    

    
    
    
### Deprecated versions compatible with gpflow version 2.5.2


class SoftmaxFixed_depracated(Softmax):
    """
    Modification to method _log_prob required.
    See issue: https://github.com/GPflow/GPflow/issues/1591
    
    Works with gpflow version 2.5.2 where data X is not required argument.
    """
    def _log_prob(self, F: TensorType, Y: TensorType) -> tf.Tensor:
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=tf.argmax(Y, axis=1))
    
    
    
    
    
class MultilatentBernoulli_deprecated(MultiLatentLikelihood):
    '''
    Mean field approximation given two or more a priori independent latent processes.
    
    Works with gpflow version 2.5.2 where data X is not required argument.
    '''
    def __init__(self, 
                 latent_dim: int,
                 invlink: Callable[[tf.Tensor], tf.Tensor] = inv_probit, 
                 **kwargs: Any
                ) -> None:
        
        super().__init__(
            latent_dim=latent_dim,
            **kwargs,
        )
        self.invlink = invlink
        
    def _conditional_mean(self, F: TensorType) -> tf.Tensor:
        '''
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :returns: invlink(mean) [..., 1]
        '''
        Fsum = tf.expand_dims(tf.reduce_sum(F, axis=-1),1) # Sum means of latent GP posteriors
        return self.invlink(Fsum)
    
    def _conditional_variance(self, F: TensorType) -> tf.Tensor:
        p = self.conditional_mean(X, F)
        return p - (p ** 2)
    
    def _predict_mean_and_var(
        self, 
        Fmu: TensorType, 
        Fvar: TensorType
    ) -> MeanAndVariance:
        '''
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        '''
        if self.invlink is inv_probit:
            FmuSum = tf.expand_dims(tf.reduce_sum(Fmu, axis=-1),1) # Sum means of latent GP posteriors
            FvarSum = tf.expand_dims(tf.reduce_sum(Fvar, axis=-1),1) # Sum variance of latent GP posteriors (only diags)
            p = inv_probit(FmuSum / tf.sqrt(1 + FvarSum))
            return p, p - tf.square(p)
        else:
            return super()._predict_mean_and_var(Fmu, Fvar)
        
    def _log_prob(self, F: TensorType, Y: TensorType) -> tf.Tensor:
        """
        The log probability density log p(Y|F)

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., 1]:
        :returns: log density Bernoulli, with shape [...]
        """
        Fsum = tf.reduce_sum(F, axis=-1, keepdims=True) # Sum means of latent GP posteriors
        return tf.squeeze(logdensities.bernoulli(Y, self.invlink(Fsum)),-1)