import torch
import torch.nn as nn
import math

__all__ = ['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l', 'efficientnetv2_xl']

# from torchsummary import summary


# 这个函数的目的是确保Channel能被8整除。
def _make_divisible(v, divisor, min_value=None):
    """
    这个函数的目的是确保Channel能被8整除。
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    """
     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """

    def __init__(self, inp, oup, stride, expand_ratio, fused):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if fused:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:

            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNetv2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EfficientNetv2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)

        layers = [conv_3x3_bn(8*3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, fused in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, fused))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def efficientnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, fused
        [1, 24, 2, 1, 1],
        [4, 48, 4, 2, 1],
        [4, 64, 4, 2, 1],
        [4, 128, 6, 2, 0],
        [6, 160, 9, 1, 0],
        [6, 272, 15, 2, 0],
    ]
    return EfficientNetv2(cfgs, **kwargs)


def efficientnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, fused
        [1, 24, 3, 1, 1],
        [4, 48, 5, 2, 1],
        [4, 80, 5, 2, 1],
        [4, 160, 7, 2, 0],
        [6, 176, 14, 1, 0],
        [6, 304, 18, 2, 0],
        [6, 512, 5, 1, 0],
    ]
    return EfficientNetv2(cfgs, **kwargs)


def efficientnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, fused
        [1, 32, 4, 1, 1],
        [4, 64, 7, 2, 1],
        [4, 96, 7, 2, 1],
        [4, 192, 10, 2, 0],
        [6, 224, 19, 1, 0],
        [6, 384, 25, 2, 0],
        [6, 640, 7, 1, 0],
    ]
    return EfficientNetv2(cfgs, **kwargs)


def efficientnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, fused
        [1, 32, 4, 1, 1],
        [4, 64, 8, 2, 1],
        [4, 96, 8, 2, 1],
        [4, 192, 16, 2, 0],
        [6, 256, 24, 1, 0],
        [6, 512, 32, 2, 0],
        [6, 640, 8, 1, 0],
    ]
    return EfficientNetv2(cfgs, **kwargs)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = efficientnetv2_s()
    model.to(device)
    # summary(model, (3, 224, 224))
