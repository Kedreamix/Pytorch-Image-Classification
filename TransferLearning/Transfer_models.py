import torchvision.models as models
import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def AlexNet(classes=1000):
    # 导入Pytorch封装的AlexNet网络模型
    alexnet = models.alexnet(pretrained=True)
    # 固定卷积层参数
    for param in alexnet.parameters():
        param.requires_grad = False
    
    # 获取最后一个全连接层的输入通道数
    num_input = alexnet.classifier[6].in_features
    
    # 获取全连接层的网络结构
    feature_model = list(alexnet.classifier.children())
    
    # 去掉原来的最后一层
    feature_model.pop()
    
    # 添加上适用于自己数据集的全连接层
    feature_model.append(nn.Linear(num_input, classes))
    # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
    # 还可以为网络添加新的层
    # 重新生成网络的后半部分
    alexnet.classifier = nn.Sequential(*feature_model)
    for param in alexnet.classifier.parameters():
        param.requires_grad = True
    alexnet = alexnet.to(device)
    # 打印一下
    # print(alexnet)
    return alexnet

def VGG16(classes=1000):
    vgg16 = models.vgg16_bn(pretrained=True)
    # 固定模型权重
    for param in vgg16.parameters():
        param.requires_grad = False
        
    # 最后加一个分类器
    vgg16.classifier[6] = nn.Sequential(nn.Linear(4096, classes))
    for param in vgg16.classifier[6].parameters():
        param.requires_grad = True
        
    vgg16 = vgg16.to(device)
    return vgg16


def ResNet18(classes=1000):
    resnet18 = models.resnet18(pretrained=True)

    for param in resnet18.parameters():
        param.requires_grad = False
        
    inchannel = resnet18.fc.in_features
    resnet18.fc = nn.Linear(inchannel, classes)
    for param in resnet18.fc.parameters():
        param.requires_grad = True
        
    resnet18 = resnet18.to(device)
    return resnet18


def DenseNet121(classes=1000):
    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False
        
    inchannel = densenet121.classifier.in_features
    densenet121.classifier = nn.Linear(inchannel, classes)
    for param in densenet121.classifier.parameters():
        param.requires_grad = True
        
    densenet121 = densenet121.to(device)
    return densenet121


def MobileNetv2(classes=1000):
    mobilenet = models.mobilenet_v2(pretrained=True)

    for param in mobilenet.parameters():
        param.requires_grad = False
        
    # 最后加一个分类器
    mobilenet.classifier[1] = nn.Sequential(nn.Linear(1280, classes))
    for param in mobilenet.classifier[1].parameters():
        param.requires_grad = True
        
    mobilenet = mobilenet.to(device)
    return mobilenet

def ShuffleNetv2(classes=1000):
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

    for param in shufflenet.parameters():
        param.requires_grad = False
        
    # 最后加一个分类器
    inchannel = shufflenet.fc.in_features
    shufflenet.fc = nn.Linear(inchannel, classes)
    for param in shufflenet.fc.parameters():
        param.requires_grad = True
        
    shufflenet = shufflenet.to(device)
    return shufflenet
    
def MnasNet(classes=1000):
    mnasnet = models.mnasnet1_0(pretrained=True)
    for param in mnasnet.parameters():
        param.requires_grad = False
        
    # 最后加一个分类器
    mnasnet.classifier[1] = nn.Sequential(nn.Dropout(p=0.2),nn.Linear(1280, classes))
    for param in mnasnet.classifier[1].parameters():
        param.requires_grad = True
        
    mnasnet = mnasnet.to(device)
    return mnasnet
