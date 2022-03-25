import torch.nn as nn
import torchvision
import os
from utils import *
from Transfer_models import *
import numpy as np
import argparse

def main(opt):
    transform = transforms.Compose([
        transforms.Resize(256),# Resize成256x256
        transforms.CenterCrop(224),# 随机裁剪为224x224
        transforms.ToTensor(),# 转化为向量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # 进行标准化
    
    root = opt.data
    # root = '../cats_and_dogs_filtered/' # 把文件夹放在当前目录下
    trainset = torchvision.datasets.ImageFolder(root + 'train',transform=transform)
    valset = torchvision.datasets.ImageFolder(root + 'validation',transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,shuffle=True,num_workers=opt.workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size,shuffle=False,num_workers=opt.workers)

    print('{} DATA has loaded'.format(root))
    train_size = len(trainset)
    val_size = len(valset)
    test_size = 1000 # 测试集文件夹下的图片数量
    print(u"训练集个数:", len(trainset))
    print(u"验证集个数:", len(valset))

    if opt.save_path != '':
        # 模型文件
        if not os.path.exists('./model/'):
            os.mkdir('./model')
    if opt.logs:
        # 日志文件，以防中断以后继续加载模型训练
        if not os.path.exists('./logs/'):
            os.mkdir('./logs')
    if opt.verbose:
        if not os.path.exists('./vis/'):
            os.mkdir('./vis')
    CLASSES = opt.classes # 类别
    if opt.model == "AlexNet": 
        model = AlexNet(CLASSES)
    elif opt.model == "VGG":model = VGG16(CLASSES)
    elif opt.model == "ResNet":model = ResNet18(CLASSES)
    elif opt.model == "DenseNet":model = DenseNet121(CLASSES)
    elif opt.model == "MobileNet":model = MobileNetv2(CLASSES)
    elif opt.model == "ShuffleNet":model = ShuffleNetv2(CLASSES)
    elif opt.model == "MnasNet":model = MnasNet(CLASSES)
    
    print("You Choose {} model".format(opt.model))
    
    #---------------------------------------#
    #  定义优化器，损失函数以及学习策略
    #---------------------------------------#
    if opt.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9,nesterov=True)
    elif opt.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
    elif opt.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr)
    print("We use {} optimizer".format(opt.optimizer))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=0.5,min_lr=1e-6)

    #-------------------------------------------------#
    #  进行训练，默认10个epochs，保存路径是model.pth, 
    #  不进行tensorboard可视化，记录损失，
    #  存在日志记录，没有预训练模型
    #-------------------------------------------------#

    if opt.verbose:
        Acc, Loss, Lr = train(model,trainloader, valloader, optimizer , criterion, scheduler , epochs = opt.epochs , path = opt.save_path , writer = False, verbose = opt.verbose, logs = opt.logs, pretrain = opt.weights)
        plot_history(opt.epochs, Acc, Loss, Lr)
        n = np.random.randint(0,test_size)
        test(root + 'test/%d.jpg'% n ,model)
    else:
        train(model,trainloader, valloader, optimizer , criterion, scheduler , epochs = opt.epochs , path = opt.save_path , verbose = opt.verbose, logs = opt.logs, pretrain = opt.weights)

if __name__ == '__main__':
    # python train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,help='epochs for training')
    parser.add_argument('--classes', type=int, default=2,help='the number of classes')
    parser.add_argument('--model', type=str,choices=['AlexNet','VGG','ResNet','MobileNet','ShuffleNet','DenseNet','MnasNet'], default='MnasNet', help='select model to train')
    parser.add_argument('--data', type=str, default='./cats_and_dogs_filtered/', help='data 文件路径') 
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--lr',type=int,default=0.02,help='Learning Rate')
    parser.add_argument('--weights',type=str,default='',help='initial weights path')
    parser.add_argument('--save-path',type=str,default='model.pth',help='save model')
    parser.add_argument('--verbose',action='store_true',help='可视化训练结果')
    parser.add_argument('--logs',action='store_true',help='每次迭代后都保存模型')
    parser.add_argument('--test-size',type=int,default=1000,help='the number for picture in data/test')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # 可以自动选择GPU，但是这里我就让程序自动判断
    
    opt = parser.parse_args()
    print(opt)
    main(opt)
