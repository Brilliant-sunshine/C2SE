import torch
import torch.nn as nn

from torch.nn.functional import elu
from torch.autograd import Function
from sklearn.metrics import hinge_loss
import numpy as np
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
import torch.nn.functional as F
from braindecode.torch_ext.util import np_to_var

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
                                        maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


F1 = 8
D = 2
F2 = 16  # usually set to F1*D (?)
third_kernel_size = (8, 4)
drop_prob = 0.25


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


class eegnet(nn.Module):
    def __init__(self, n_classes, in_chans, input_time_length, kernel_length, final_conv_length='auto'):
        super(eegnet, self).__init__()
        self.kernel_length = kernel_length
        self.final_conv_length = final_conv_length
        self.in_chans = in_chans
        self.input_time_length = input_time_length
        self.n_classes = n_classes
        self.batch_norm = True

        self.feature = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(1, F1, (1, self.kernel_length), stride=1, bias=False, padding=(0, self.kernel_length // 2,)),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(F1, F1 * D, (self.in_chans, 1), max_norm=1, stride=1, bias=False, groups=F1,
                                 padding=(0, 0)),
            # nn.Conv2d(F1 , F1 * D, (self.in_chans, 1), stride=1, bias=False, padding=(0, 0)),

            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            Expression(elu),

            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=drop_prob),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), stride=1, bias=False, groups=F1 * D, padding=(0, 16 // 2)),
            nn.Conv2d(F1 * D, F2, (1, 1), stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            Expression(elu),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=drop_prob),
        )
        if self.final_conv_length == 'auto':
            out = self.feature(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length, 1),
                dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.classifier = torch.nn.Linear(F2 * self.final_conv_length,
                                          self.n_classes, bias=True)


    def forward(self, x):
        feature1 = self.feature(x)

        x = feature1.squeeze()
        feature = x.reshape(x.size(0), -1)
        output = self.classifier(feature)

        return output, feature1
