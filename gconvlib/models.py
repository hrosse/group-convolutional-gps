import numpy as np
import tensorflow as tf

from gpflow.base import MeanAndVariance, InputData
from gpflow.models import SVGP
from gpflow import posteriors

class SVGP_with_min_var(SVGP):
    
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Modification: Clip variance to a small positive constant from below.
        It can happen that during training the variance of q(f(x)) becomes slightly negative when using group-convolutional GPs.
        Particularly when using float32.
        Then ELBO becomes NaN and the parameters of the model become Nan when updating.
        This typically happens when loss function starts to change only very little.
        Even when skipping update step during optimisation this happens with increasing frequency.
        """
        fmean, fvar = self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return fmean, tf.maximum(fvar, 1e-5)