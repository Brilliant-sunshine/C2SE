import numpy as np
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var

class Deep4Net_base(nn.Module):
    """
    Deep ConvNet model from [2]_.

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
        super(Deep4Net_base, self).__init__()        

        self.in_chans = in_chans
        self.input_time_length = input_time_length
        self.n_classes = n_classes
        self.final_conv_length = final_conv_length
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        
        pool_stride = 3
        self.features = nn.Sequential()

        self.features.add_module('dimshuffle', Expression(_transpose_time_to_spat))
        self.features.add_module('conv_time', nn.Conv2d(1, 25,
                                                (10, 1),
                                                stride=1, ))
        self.features.add_module('conv_spat',
                         nn.Conv2d(25, 25,
                                   (1, self.in_chans),
                                   stride=(1, 1),
                                   bias=not self.batch_norm))
        n_filters_conv = 25

        if self.batch_norm:
            self.features.add_module('bnorm',
                             nn.BatchNorm2d(n_filters_conv,
                                            momentum=0.1,
                                            affine=True,
                                            eps=1e-5),)
        self.features.add_module('conv_nonlin', Expression(elu))
        self.features.add_module('pool',
                         nn.MaxPool2d(
                             kernel_size=(3, 1),
                             stride=(3, 1)))
        self.features.add_module('pool_nonlin', Expression(identity))

        def add_conv_pool_block(model, n_filters_before,
                                n_filters, filter_length, block_nr):
            suffix = '_{:d}'.format(block_nr)
            model.add_module('drop' + suffix,
                             nn.Dropout(p=self.drop_prob))
            model.add_module('conv' + suffix,
                             nn.Conv2d(n_filters_before, n_filters,
                                       (filter_length, 1),
                                       stride=(1, 1),
                                       bias=not self.batch_norm))
            if self.batch_norm:
                model.add_module('bnorm' + suffix,
                             nn.BatchNorm2d(n_filters,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5))
            model.add_module('nonlin' + suffix,
                             Expression(elu))

            model.add_module('pool' + suffix,
                             nn.MaxPool2d(
                                 kernel_size=(3, 1),
                                 stride=(pool_stride, 1)))
            model.add_module('pool_nonlin' + suffix,
                             Expression(identity))

        add_conv_pool_block(self.features, n_filters_conv, 50, 10, 2)
        add_conv_pool_block(self.features, 50, 100, 10, 3)
        add_conv_pool_block(self.features, 100, 200, 10, 4)

        if self.final_conv_length == 'auto':
            out = self.features(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length,1),
                dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('conv_classifier',
                             nn.Conv2d(200, self.n_classes,
                                       (self.final_conv_length, 1), bias=True))
        self.classifier.add_module('squeeze',  Expression(_squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.features.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm

        init.constant_(self.features.conv_time.bias, 0)

        init.xavier_uniform_(self.features.conv_spat.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.features.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.features.bnorm.weight, 1)
            init.constant_(self.features.bnorm.bias, 0)
        param_dict = dict(list(self.features.named_parameters()))
        for block_nr in range(2,5):
            conv_weight = param_dict['conv_{:d}.weight'.format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict['conv_{:d}.bias'.format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict['bnorm_{:d}.weight'.format(block_nr)]
                bnorm_bias = param_dict['bnorm_{:d}.bias'.format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.classifier.conv_classifier.weight, gain=1)
        init.constant_(self.classifier.conv_classifier.bias, 0)

    def forward(self, x):
        feature = self.features(x)  
  
        out = self.classifier(feature)
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

