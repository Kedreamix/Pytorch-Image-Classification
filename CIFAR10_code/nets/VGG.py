'''
VGG11,13,16,19 in pytorch
'''

from turtle import forward
import torch
import torch.nn as nn
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vggname = 'VGG16',num_classes=10):
        super(VGG,self).__init__()
        self.features = self._make_layers(cfg[vggname])
        self.classifier = nn.Linear(512,num_classes)
    
    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M': # 最大池化层
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [nn.Conv2d(in_channels,out_channels=x,kernel_size=3,padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x
    
def test():
    net = VGG('VGG19')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(2,3,32,32))
    
test()