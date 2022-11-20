'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from utils import get_acc,EarlyStopping
from dataloader import get_test_dataloader, get_training_dataloader
from tqdm import tqdm


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help =' use GPU?')
    parser.add_argument('--batch-size', default=64, type=int, help = "Batch Size for Training")
    parser.add_argument('--num-workers', default=2, type=int, help = 'num-workers')
    parser.add_argument('--net', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19','ResNet18','ResNet34',   
                                                       'DenseNet','MobileNetv1','MobileNetv2'], default='MobileNetv1', help='net type')
    parser.add_argument('--epochs', type = int, default=20, help = 'Epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--patience', '-p', type = int, default=7, help='patience for Early stop')
    parser.add_argument('--optim','-o',type = str, choices = ['sgd','adam','adamw'], default = 'adamw', help = 'choose optimizer')

    args = parser.parse_args()
    
    print(args)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Train Data
    trainloader = get_training_dataloader(batch_size = args.batch_size, num_workers = args.num_workers)
    testloader = get_test_dataloader(batch_size = args.batch_size, num_workers = args.num_workers, shuffle=False)
    # Model
    print('==> Building model..')
    if args.net == 'VGG16':
        from nets.VGG import VGG
        net = VGG('VGG16')
    elif args.net == 'VGG19':
        from nets.VGG import VGG
        net = VGG('VGG19')
    elif args.net == 'ResNet18':
        from nets.ResNet import ResNet18
        net = ResNet18()
    elif args.net == 'ResNet34':
        from nets.ResNet import ResNet34
        net = ResNet34()
    elif args.net == 'LeNet5':
        from nets.LeNet5 import LeNet5
        net = LeNet5()
    elif args.net == 'AlexNet':
        from nets.AlexNet import AlexNet
        net = AlexNet()
    elif args.net == 'DenseNet':
        from nets.DenseNet import densenet_cifar
        net = densenet_cifar()
    elif args.net == 'MobileNetv1':
        from nets.MobileNetv1 import MobileNet
        net = MobileNet()
    elif args.net == 'MobileNetv2':
        from nets.MobileNetv2 import MobileNetV2
        net = MobileNetV2()

    if args.cuda:
        device = 'cuda'
        net = torch.nn.DataParallel(net)
        # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        args.lr = checkpoint['lr']

    early_stopping = EarlyStopping(patience = args.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94,verbose=True,patience = 1,min_lr = 0.000001) # 动态更新学习率

    epochs = args.epochs
    def train(epoch):
        epoch_step = len(trainloader)
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
        net.train()
        train_loss = 0
        train_acc = 0
        print('Start Train')
        with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,(im,label) in enumerate(trainloader,start=0):
                im = im.to(device)
                label = label.to(device)
                #---------------------
                #  释放内存
                #---------------------
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播forward
                #----------------------#
                outputs = net(im)
                #----------------------#
                #   计算损失
                #----------------------#
                loss = criterion(outputs,label)
                train_loss += loss.data
                train_acc += get_acc(outputs,label)
                #----------------------#
                #   反向传播
                #----------------------#
                # backward
                loss.backward()
                # 更新参数
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(**{'Train Loss' : train_loss.item()/(step+1),
                                    'Train Acc' :train_acc.item()/(step+1),  
                                    'Lr'   : lr})
                pbar.update(1)
        # train_loss = train_loss.item() / len(trainloader)
        # train_acc = train_acc.item() * 100 / len(trainloader)
        scheduler.step(train_loss)
        print('Finish Train')
    def test(epoch):
        global best_acc
        epoch_step_test = len(testloader)
        if epoch_step_test == 0:
                raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
        
        net.eval()
        test_loss = 0
        test_acc = 0
        print('Start Test')
        #--------------------------------
        #   相同方法，同train
        #--------------------------------
        with tqdm(total=epoch_step_test,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar2:
            for step,(im,label) in enumerate(testloader,start=0):
                im = im.to(device)
                label = label.to(device)
                with torch.no_grad():
                    if step >= epoch_step_test:
                        break
                    
                    # 释放内存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs = net(im)
                    loss = criterion(outputs,label)
                    test_loss += loss.data
                    test_acc += get_acc(outputs,label)
                    
                    pbar2.set_postfix(**{'Test Acc': test_acc.item()/(step+1),
                                'Test Loss': test_loss.item() / (step + 1)})
                    pbar2.update(1)
        lr = optimizer.param_groups[0]['lr']
        test_acc = test_acc.item() * 100 / len(testloader)
        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'lr': lr,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}_ckpt.pth'.format(args.net))
            best_acc = test_acc
            
        print('Finish Test')

        early_stopping(test_loss, net)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            exit()
        
    for epoch in range(start_epoch, epochs):
        train(epoch)
        test(epoch)
        
    torch.cuda.empty_cache()
    