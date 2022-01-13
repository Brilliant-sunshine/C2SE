import numpy as np
from torch import nn
from torch.nn import init
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var


class ShallowFBCSPNet(nn.Module):

    def __init__(self, in_chans, n_classes, input_time_length, final_conv_length='auto'):
        super(ShallowFBCSPNet, self).__init__()
        self.final_conv_length = final_conv_length
        self.in_chans = in_chans
        self.input_time_length = input_time_length
        self.n_classes = n_classes
        self.batch_norm = True

        self.features = nn.Sequential()
        self.features.add_module('dimshuffle', Expression(_transpose_time_to_spat))
        self.features.add_module('conv_time', nn.Conv2d(1, 40, (25, 1), stride=(1, 1), ))
        self.features.add_module('conv_spat', nn.Conv2d(40, 40, (1, self.in_chans), stride=1, bias=not self.batch_norm))
        n_filters_conv = 40

        self.features.add_module('bnorm', nn.BatchNorm2d(n_filters_conv, momentum=0.1, affine=True), )

        self.features.add_module('conv_nonlin', Expression(square))
        self.features.add_module('pool', nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1)))  # $$$$ 15->10
        self.features.add_module('pool_nonlin', Expression(safe_log))
        # self.features.add_module('drop', nn.Dropout(p=0.2))

        if self.final_conv_length == 'auto':
            out = self.features(np_to_var(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        self.linear = nn.Linear(n_filters_conv * self.final_conv_length, 128, bias=True)

        self.classifier_1 = nn.Sequential()
        self.classifier_1.add_module('fc1', nn.Linear(n_filters_conv * self.final_conv_length, n_classes, bias=True))
        # self.classifier_1.add_module('fc1', nn.Linear(128, 4, bias=True))

        # self.classifier_1.add_module("softmax", nn.LogSoftmax(dim=1))
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

        init.xavier_uniform_(self.classifier_1.fc1.weight, gain=1)
        init.constant_(self.classifier_1.fc1.bias, 0)
        # init.xavier_uniform_(self.classifier_1.fc2.weight, gain=1)
        # init.constant_(self.classifier_1.fc2.bias, 0)

        init.xavier_uniform_(self.linear.weight, gain=1)
        init.constant_(self.linear.bias, 0)

        # con = nn.Conv2d(1, 40, (25, 1), stride=(1, 1), )

    def forward(self, x):
        feature = self.features(x.float())

        f1 = feature.view(feature.shape[0], -1)
        f2 = self.linear(f1)
        out = self.classifier_1(f1)

        return out  # , f2, f1


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)
