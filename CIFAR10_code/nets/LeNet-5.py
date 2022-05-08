'''
LetNet in Pytorch
'''
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes = 10, init_weights=True):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Sequential(
            # 输入 32x32x3 -> 28x28x6 (32-5)/1 + 1=28
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1),
            nn.ReLU(),
            # 经过2x2的maxpool，变成14x14 (28-2)/2+1
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.conv2 = nn.Sequential(
            # 输入 14x14x6 ->  10x10x16    (14-5)/1 + 1 = 10
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            nn.ReLU(),
            # (10-2)/2 + 1 = 5 
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(5*5*16,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,num_classes)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 要把多维度的tensor展平成一维
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    

def test():
    net = LeNet5()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(2,3,32,32))
    
