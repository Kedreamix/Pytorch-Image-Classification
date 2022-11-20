'''
MobileNetv1 in pytorch

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self,in_channels,out_channels,stride=1):
        super(Block,self).__init__()
        # groups参数就是深度可分离卷积的关键
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,
                               padding=1,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
        
# 深度可分离卷积 DepthWise Separable Convolution
class MobileNet(nn.Module):
    # (128,2) means conv channel=128, conv stride=2, by default conv stride=1
    cfg = [64,(128,2),128,(256,2),256,(512,2),512,512,512,512,512,(1024,2),1024]
    
    def __init__(self, num_classes=10,alpha=1.0,beta=1.0,init_weights=True):
        super(MobileNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.avg = nn.AvgPool2d(kernel_size=2)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(1024,num_classes)
    
    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x,int) else x[0]
            stride = 1 if isinstance(x,int) else x[1]
            layers.append(Block(in_channels,out_channels,stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avg(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
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


def test():
    net = MobileNet()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(2,3,32,32))
    
# test()