from gpflow.inducing_variables import InducingPoints


# used for inter-domain inducing variables with G-Kernel.
class InducingImages(InducingPoints):
    '''
    For inter-domain inducing variables with G-Kernel.
    '''
    pass


# not used
class InducingPatchesLoopX(InducingPoints):
    '''
    For inter-domain kernel instances of group-convolutional GPs with loop.
    '''
    pass