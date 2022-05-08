'''
MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn


class Block(nn.Module):
    # 使用了inverted residuals
    '''expand + depthwise + pointwise'''
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        channels        = expansion * in_channels  # 倒残差结构先升维 再降维
        self.conv1      = nn.Conv2d(in_channels, channels, kernel_size = 1, stride=1, padding=0, bias=False)
        self.bn1        = nn.BatchNorm2d(channels)
        self.conv2      = nn.Conv2d(channels,channels,kernel_size=3,stride=stride,padding=1, groups=channels, bias=False)
        self.bn2        = nn.BatchNorm2d(channels)
        self.conv3      = nn.Conv2d(channels, out_channels, kernel_size=1,stride=1, padding = 0, bias=False)
        self.bn3        = nn.BatchNorm2d(out_channels)
        
        self.shortcut   = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu6      = nn.ReLU6()    
    def forward(self, x):
        out = self.relu6(self.bn1(self.conv1(x)))
        out = self.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        
        return out
    
class MobileNetV2(nn.Module):
    # (expansion, out_channels, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1280, num_classes)
        self.relu6 = nn.ReLU6()
        
    def _make_layers(self, in_channels):
        layers = []
        for expansion, out_channels, num_block, stride in self.cfg:
            strides = [stride] + [1]*(num_block-1)
            for stride in strides:
                layers.append(Block(in_channels, out_channels, expansion, stride))
                in_channels = out_channels
        return nn.Sequential(*layers)
    
    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)    
    
    def forward(self, x):
        out = self.relu6(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.relu6(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out

    
def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(32,3,32,32))
    
test() 