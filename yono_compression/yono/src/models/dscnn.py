"""
    DS-CNN for ImageNet-1K, implemented in PyTorch.
    Original paper: 'DS-CNN: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.
"""

__all__ = ['PTCVDSCNN','DSCNN', 'dscnn_l', 'dscnn_m', 'dscnn_s', 'micronet_l', 'micronet_m', 'micronet_s', 'get_dscnn']

import os
import torch.nn as nn
from pytorchcv.models.common import dwsconv3x3_block, ConvBlock
import torch.nn.functional as F


class PTCVDSCNN(nn.Module):
    def __init__(self, model_name="dscnn_l", in_channels=1, num_classes=12, 
     init_block_kernel=(10,4), init_block_channel=192, init_block_stride=(1,1),
     pretrained=False, **kwargs):
        super(PTCVDSCNN, self).__init__()
        self.model = get_dscnn(model_name=model_name.lower(),
                               in_channels=in_channels,
                               num_classes=num_classes,
                               init_block_kernel=init_block_kernel,
                               init_block_channel=init_block_channel,
                               init_block_stride=init_block_stride,
                               pretrained=pretrained,
                               **kwargs)
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)


# padding has either 1 equivalent number or 4 numbers [left, right, top, bottom]
def get_same_padding(kernel_size=(10,4)):
    # when kernel size is the same and odd number
    # return 1 equivalent number as a padding
    if (kernel_size[0] == kernel_size[1]) and (kernel_size[0] % 2 == 1):
        return int((kernel_size[0]-1)/2)
    # when kernel size differs and contains even number
    # return 4 numbers as a padding
    else:
        # kernel_size == (Height, Weight)
        # Height == (top, bottom)
        # Weight == (left, right)
        padding = [0,0,0,0]
        # Kernel Height => padding's top bottom, i.e., padding[2,3]
        if kernel_size[0] % 2 == 1:
            padding[2], padding[3] = int((kernel_size[0]-1)/2), int((kernel_size[0]-1)/2)
        else:
            padding[2], padding[3] = int(kernel_size[0] / 2) - 1, int(kernel_size[0] / 2)
        # Kernel Width => padding's left right, i.e., padding[0,1]
        if kernel_size[1] % 2 == 1:
            padding[0], padding[1] = int((kernel_size[1]-1)/2), int((kernel_size[1]-1)/2)
        else:
            padding[0], padding[1] = int(kernel_size[1] / 2) - 1, int(kernel_size[1] / 2)
        return padding

class DSCNN(nn.Module):
    """
    DS-CNN model from 'DS-CNNs: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_stage_stride : bool
        Whether stride is used at the first stage.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 init_block_channel,
                 init_block_kernel,
                 init_block_stride,
                 channels,
                 strides,
                 dw_use_bn=True,
                 dw_activation=(lambda: nn.ReLU(inplace=True)),
                 in_channels=1,
                 num_classes=12):
        super(DSCNN, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channel,
            kernel_size=init_block_kernel,
            stride=init_block_stride,
            padding=get_same_padding(init_block_kernel))
            )

        in_channels = init_block_channel
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = strides[i]
                stage.add_module("unit{}".format(j + 1), dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dw_use_bn=dw_use_bn,
                    dw_activation=dw_activation))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        #### Note: Add dropout layer after all blocks and before pooling
        self.features.add_module("final_dropout", nn.Dropout(0.4))
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d((1, 1)))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in')
            elif name == 'init_block.conv' or 'pw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif 'bn' in name:
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif 'output' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_dscnn(model_name=None,
              in_channels=1,
              num_classes=12,
              init_block_kernel=(10,4),
              init_block_channel=192,
              init_block_stride=(1,1),
              pretrained=False,
              dws_simplified=False,
              width_scale=1.0,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create DS-CNN model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    dws_simplified : bool, default False
        Whether to use simplified depthwise separable convolution block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    # config for MobileNetV1
    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    # config for DS-CNN for KWS
    if "dscnn_l" in model_name or "dscnn-l" in model_name or "ds-cnn_l" in model_name or "ds-cnn-l" in model_name:
        init_block_channel = 276
        init_block_stride = (2,1)
        channels = [[276],[276],[276],[276],[276]]
        strides = [2,1,1,1,1]
    elif "dscnn_m" in model_name or "dscnn-m" in model_name or "ds-cnn_m" in model_name or "ds-cnn-m" in model_name:
        init_block_channel = 172
        init_block_stride = (2,1)
        channels = [[172],[172],[172],[172]]
        strides = [2, 1, 1, 1]
    elif "dscnn_s" in model_name or "dscnn-s" in model_name or "ds-cnn_s" in model_name or "ds-cnn-s" in model_name:
        init_block_channel = 64
        init_block_stride = (2,2)
        channels = [[64],[64],[64],[64]]
        strides = [1, 1, 1, 1]
    # config for MicroNet for KWS
    elif "micronet_l" in model_name or "micronet-l" in model_name:
        init_block_channel = 276
        init_block_stride = (1,1)
        channels = [[248],[276],[276],[248],[248],[248],[248]]
        strides = [2, 1, 1, 1, 1, 1, 1]
    elif "micronet_ad_l" in model_name or "micronet-ad-l" in model_name:
        init_block_channel = 276
        init_block_stride = (1,1)
        channels = [[248],[276],[276],[248],[248]]
        strides = [2, 1, 1, 2, 2]
    elif "micronet_m" in model_name or "micronet-m" in model_name:
        init_block_channel = 140
        init_block_stride = (1,1)
        channels = [[140],[140],[140],[112],[196]]
        strides = [2, 1, 1, 1, 1]
    elif "micronet_ad_m" in model_name or "micronet-ad-m" in model_name:
        init_block_channel = 192
        init_block_stride = (1,1)
        channels = [[276],[276],[276],[276],[276]]
        strides = [2, 1, 1, 2, 2]
    elif "micronet_ad_k" in model_name or "micronet-ad-k" in model_name:
        init_block_channel = 192
        init_block_stride = (1,1)
        channels = [[192],[192],[192],[192],[192]]
        strides = [2, 1, 1, 2, 2]
    elif "micronet_ad_i160" in model_name or "micronet-ad-i160" in model_name:
        init_block_channel = 160
        init_block_stride = (2,2)
        channels = [[192],[192],[192],[192],[192]]
        strides = [2, 1, 1, 2, 2]
    elif "micronet_ad_i160" in model_name or "micronet-ad-i160" in model_name:
        init_block_channel = 160
        init_block_stride = (1,1)
        channels = [[192],[192],[192],[192],[192]]
        strides = [2, 1, 1, 2, 2]
    elif "micronet_ad_i176" in model_name or "micronet-ad-i176" in model_name:
        init_block_channel = 176
        init_block_stride = (1,1)
        channels = [[192],[192],[192],[192],[192]]
        strides = [2, 1, 1, 2, 2]
    elif "micronet_ad_i" in model_name or "micronet-ad-i" in model_name:
        init_block_channel = 112
        init_block_stride = (1,1)
        channels = [[192],[192],[192],[192],[192]]
        strides = [2, 1, 1, 2, 2]
    elif "micronet_s" in model_name or "micronet-s" in model_name:
        init_block_channel = 84
        init_block_stride = (1,1)
        channels = [[112],[84],[84],[84],[196]]
        strides = [2, 1, 1, 1, 1]
    elif "hhar" in model_name or "skoda" in model_name:
        init_block_channel = 176
        # init_block_stride = (2,2)
        init_block_stride = (4,1)
        channels = [[192],[192],[192],[192],[192]]
        strides = [2, 1, 1, 2, 2]
    elif "pamap2" in model_name:
        init_block_channel = 176
        init_block_stride = (2,1)
        channels = [[192],[192],[192],[192],[192]]
        strides = [2, 1, 1, 2, 2]

    if "ch168" in model_name:
        init_block_channel = 168
    elif "ch172" in model_name:
        init_block_channel = 172
    elif "ch176" in model_name:
        init_block_channel = 176

    if 's22' in model_name:
        init_block_stride = (2,2)
    elif 's33' in model_name:
        init_block_stride = (3,3)
    elif 's21' in model_name:
        init_block_stride = (2,1)
    elif 's12' in model_name:
        init_block_stride = (1,2)
    elif 's41' in model_name:
        init_block_stride = (4,1)
    elif 's14' in model_name:
        init_block_stride = (1,4)

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    if dws_simplified:
        dw_use_bn = False
        dw_activation = None
    else:
        dw_use_bn = True
        dw_activation = (lambda: nn.ReLU(inplace=True))

    net = DSCNN(
        init_block_channel=init_block_channel,
        init_block_kernel=init_block_kernel,
        init_block_stride=init_block_stride,
        channels=channels,
        strides=strides,
        dw_use_bn=dw_use_bn,
        dw_activation=dw_activation,
        in_channels=in_channels,
        num_classes=num_classes
    )

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from pytorchcv.models.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def dscnn_l(**kwargs):
    """
    1.0 DS-CNN-224 model from 'DS-CNNs: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dscnn(model_name="dscnn_l", in_channels=1, num_classes=12, init_block_kernel=(10,4), pretrained=False, **kwargs)


def dscnn_m(**kwargs):
    """
    0.75 DS-CNN-224 model from 'DS-CNNs: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dscnn(model_name="dscnn_m", in_channels=1, num_classes=12, init_block_kernel=(10,4), pretrained=False, **kwargs)


def dscnn_s(**kwargs):
    """
    0.5 DS-CNN-224 model from 'DS-CNNs: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dscnn(model_name="dscnn_s", in_channels=1, num_classes=12, init_block_kernel=(10,4), pretrained=False, **kwargs)



def micronet_l(**kwargs):
    """
    1.0 DS-CNN-224 model from 'DS-CNNs: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dscnn(model_name="micronet_l", in_channels=1, num_classes=12, init_block_kernel=(10,4), pretrained=False, **kwargs)


def micronet_m(**kwargs):
    """
    0.75 DS-CNN-224 model from 'DS-CNNs: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dscnn(model_name="micronet_m", in_channels=1, num_classes=12, init_block_kernel=(10,4), pretrained=False, **kwargs)


def micronet_s(**kwargs):
    """
    0.5 DS-CNN-224 model from 'DS-CNNs: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dscnn(model_name="micronet_s", in_channels=1, num_classes=12, init_block_kernel=(10,4), pretrained=False, **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        dscnn_l,
        dscnn_m,
        dscnn_s,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dscnn_l or weight_count == 4231976)
        assert (model != dscnn_m or weight_count == 2585560)
        assert (model != dscnn_s or weight_count == 1331592)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
