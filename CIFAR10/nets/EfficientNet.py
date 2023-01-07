'''
EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 激活函数
def swish(x):
    return x * x.sigmoid()

# DropConnect是一种正则化方法，它在训练过程中随机将网络中的某些权重设置为0
# DropConnect是对网络中每一个权重进行随机设置的
def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio) 
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

# SE模块实现
class SE(nn.Module):
    '''Squeeze-and-Excitation MBconv with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        # 这里的1x1的卷积核与全连接层是等价的
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True) 
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1)) # 平均池化
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out # 进行连接相乘
        return out

# MBconv模块
class MBconv(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.25, # 也就是1/4
                 drop_rate=0.):
        super(MBconv, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion 第一个升维的 1x1 卷积层
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers SE模块
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        # 仅当输入 MBConv 结构的特征矩阵与输出的特征矩阵 shape 相同时才存在 shortcut 连接
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # 当n = 1的时候，是不要第一个升维的1x1的卷积层
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0: # 有shortcut的时候用Dropout
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out

# 定义EfficientNet方法
class EfficientNet(nn.Module):
    # 默认识别的类别为10
    def __init__(self,
                 width_coefficient=1.0,
                 depth_coefficient=1.0,
                 dropout_rate = 0.2,
                 num_classes=10):
        super(EfficientNet, self).__init__()
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        cfg = {
            'num_blocks': [1, 2, 2, 3, 3, 4, 1], # repeats
            'expansion': [1, 6, 6, 6, 6, 6, 6], # expansion
            'out_channels': [16, 24, 40, 80, 112, 192, 320], # out_channels
            'kernel_size': [3, 3, 5, 3, 5, 5, 3], # kernel_size
            'stride': [1, 2, 2, 2, 1, 2, 1], # stride
            'dropout_rate': dropout_rate,
            'drop_connect_rate': 0.2,
        }
        # 配置文件
        self.cfg = cfg
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        # 第一个3x3的卷积层
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['out_channels'][-1]*int(self.width_coefficient), num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size', 'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            import math
            num_blocks = math.floor(self.depth_coefficient) * num_blocks
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    MBconv(in_channels,
                          out_channels*int(self.width_coefficient),
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels*int(self.width_coefficient)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out

def EfficientNetB0(num_classes = 10):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        num_classes=num_classes)

def EfficientNetB1(num_classes = 10):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        num_classes=num_classes)

def EfficientNetB2(num_classes = 10):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)

def EfficientNetB3(num_classes = 10):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)
    
def EfficientNetB4(num_classes = 10):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)
    
def EfficientNetB5(num_classes = 10):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)

def EfficientNetB6(num_classes = 10):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)

def EfficientNetB7(num_classes = 10):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def test():
    net = EfficientNetB0()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()