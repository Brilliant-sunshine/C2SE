import numpy as np
from torch import nn
from torch.nn import init

from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var
from torch.nn.functional import elu

class shallow_fbcsp(nn.Module):
    """
    Shallow ConvNet model from [2]_.
    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(self,in_chans=22, n_classes=4, input_time_length=1125, 
                 batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5,
                 final_conv_length = 'auto'):
        super(shallow_fbcsp, self).__init__()        
        self.final_conv_length = final_conv_length
        self.in_chans=in_chans
        self.input_time_length=input_time_length
        self.n_classes=n_classes
        self.batch_norm=batch_norm
               
        self.features = nn.Sequential()
        self.features.add_module('dimshuffle', Expression(_transpose_time_to_spat))
        self.features.add_module('conv_time', nn.Conv2d(1, 40,(25, 1),
                                                stride=(1,1), ))                                            
        self.features.add_module('conv_spat',
                         nn.Conv2d(40, 40, (1, self.in_chans), stride=1,
                                   bias=not self.batch_norm))
        n_filters_conv = 40

        self.features.add_module('bnorm',
                         nn.BatchNorm2d(n_filters_conv,
                                        momentum=0.1,
                                        affine=True),)
        
        self.pool_ac = nn.Sequential()        
        self.pool_ac.add_module('conv_nonlin', Expression(square))
        self.pool_ac.add_module('pool',
                         nn.AvgPool2d(kernel_size=(75, 1),
                                    stride=(15, 1)))
        self.pool_ac.add_module('pool_nonlin', Expression(safe_log))
        self.pool_ac.add_module('drop', nn.Dropout(p=0.5))
        
        
        if self.final_conv_length == 'auto':
            out = self.pool_ac(self.features(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length,1),
                dtype=np.float32))
            ))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
            
            
        self.classifier = nn.Sequential()
        self.classifier.add_module('conv_classifier',
                              nn.Conv2d(n_filters_conv, self.n_classes,
                                        (self.final_conv_length, 1), bias=True))

        self.classifier.add_module('squeeze',  Expression(_squeeze_final_output))


        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.features.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        init.constant_(self.features.conv_time.bias, 0)
        
        init.xavier_uniform_(self.features.conv_spat.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.features.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.features.bnorm.weight, 1)
            init.constant_(self.features.bnorm.bias, 0)
        init.xavier_uniform_(self.classifier.conv_classifier.weight, gain=1)
        init.constant_(self.classifier.conv_classifier.bias, 0)
      
    def forward(self, x):
        feature = self.features(x)  
        out = self.pool_ac(feature)    
        out = self.classifier(out)
        return out
# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:,:,:,0]
    if x.size()[2] == 1:
        x = x[:,:,0]
    return x
def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)