import torch
import torch.nn as nn

"""
    Classes and functions required for Set encoding in adaptation networks. Many of the ideas and classes here are 
    closely related to DeepSets (https://arxiv.org/abs/1703.06114).
"""


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


class SetEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self):
        super(SetEncoder, self).__init__()
        self.pre_pooling_fn = SimplePrePoolNet()
        self.pooling_fn = mean_pooling
        self.post_pooling_fn = Identity()

    def forward(self, x):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:

                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.

        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x = self.pre_pooling_fn(x)
        x = self.pooling_fn(x)
        x = self.post_pooling_fn(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """
    def __init__(self):
        super(SimplePrePoolNet, self).__init__()
        self.layer1 = self._make_conv2d_layer(3, 64)
        self.layer2 = self._make_conv2d_layer(64, 64)
        self.layer3 = self._make_conv2d_layer(64, 64)
        self.layer4 = self._make_conv2d_layer(64, 64)
        self.layer5 = self._make_conv2d_layer(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64
