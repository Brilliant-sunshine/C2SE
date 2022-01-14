import torch as th
from torch import nn


class MaxNormDefaultConstraint(object):
    """
    Applies max L2 norm 2 to the weights until the final layer and L2 norm 0.5
    to the weights of the final layer as done in [1]_.
    
    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    
    """
    def apply(self, model):
        last_weight = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
               m.weight.data = th.renorm(m.weight.data,2,0,maxnorm=0.5)
#            elif isinstance(m, nn.Conv3d):
#                m.weight.data = th.renorm(m.weight.data,2,0,maxnorm=2)
            elif isinstance(m, nn.Conv2d):
                m.weight.data = th.renorm(m.weight.data,2,0,maxnorm=2)
                last_weight = m.weight
        if last_weight is not None:
            last_weight.data = th.renorm(last_weight.data,2,0,maxnorm=0.5) 

