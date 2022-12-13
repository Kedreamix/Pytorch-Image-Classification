# Pytorch CIFAR10图像分类 工具函数utils更新v2.0篇

[toc]

这里贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)

对于上一版的工具函数utils.py，我认为可能来说，可视化的感觉还是不是很好，所以我就修改了一下我新的训练函数，为了兼容，参数基本相同，但是加入了tqdm来可视化进度条，这样也会更加的好看和直观，并且统一了一些代码的格式，使得代码稍微好看点，之前有时候有点乱，除此为了兼容一些情况，修改了部分代码，但是意义相同。

如果想看上一版的工具函数utils.py，可以查看这篇博客。[PytorchCIFAR10图像分类 工具函数utils篇](https://redamancy.blog.csdn.net/article/details/121589217)



## 设置随机种子（保证可复现性）

为了保证训练结果的可复现性，同时加了设置随机种子的代码

```python
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True #for accelerating the running
```



## 得到分类的精度（适配高版本）

首先对于这个函数来说，我变化了一下最后一行的

```python
return correct / total
```

因为这一行在多数时候是正确的，但是在一些版本的torch中是不兼容的，这时候需要使用torch.div或者torch.true_div

```python
def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1) # 得到概率
    correct = (pred_y == label).sum().data
    return torch.div(correct, total)
```



## 训练函数最新版本（增加tensorboard函数和tqdm可视化进度条）

实际上，为了保证版本的兼容，所以参数都是一样的，不同的地方是，我利用了tqdm来可视化我们的进度条

这样以后，我们能更好的看到自己的训练进度和时间，并且在训练过程中就可以看到自己的结果，这样就更好的保证了可视化，在训练过程中也可以看到准确率和损失的变化。

**参数介绍**

- **net : 所选的网络模型**
- **trainloader： 训练集加载器**
- **testloader： 测试集加载器**
- **epoches：训练次数**
- **optimizer：优化器**
- **criterion：损失函数**
- **writer：是否使用tensorboard可视化，默认为None**
- **verbose：是否使用记录准确率，损失值，学习率，默认为True**
- **scheduler：学习率调整策略**
- **path：保存迭代次数中最优的模型的权重，默认为model.pth**

```python
def train(net, trainloader, testloader, epochs, optimizer , criterion, scheduler , path = './model.pth', writer = None ,verbose = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    train_acc_list, test_acc_list = [],[]
    train_loss_list, test_loss_list = [],[]
    lr_list  = []
    for i in range(epochs):
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        if torch.cuda.is_available():
            net = net.to(device)
        net.train()
        train_step = len(trainloader)
        with tqdm(total=train_step,desc=f'Train Epoch {i + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,data in enumerate(trainloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                # 释放内存
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # formard
                outputs = net(im)
                loss = criterion(outputs,label)
                # backward
                loss.backward()
                # 更新参数
                optimizer.step()
                # 累计损失
                train_loss += loss.item()
                train_acc += get_acc(outputs,label).item()
                pbar.set_postfix(**{'Train Acc' : train_acc/(step+1),
                                'Train Loss' :train_loss/(step+1)})
                pbar.update(1)
            pbar.close()
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        # 更新学习率
        scheduler.step(train_loss)
        if testloader is not None:
            net.eval()
            test_step = len(testloader)
            with torch.no_grad():
                with tqdm(total=test_step,desc=f'Test Epoch {i + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
                    for step,data in enumerate(testloader,start=0):
                        im,label = data
                        im = im.to(device)
                        label = label.to(device)
                        # 释放内存
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        outputs = net(im)
                        loss = criterion(outputs,label)
                        test_loss += loss.item()
                        test_acc += get_acc(outputs,label).item()
                        pbar.set_postfix(**{'Test Acc' : test_acc/(step+1),
                                'Test Loss' :test_loss/(step+1)})
                        pbar.update(1)
                    pbar.close()
                test_loss = test_loss / len(testloader)
                test_acc = test_acc * 100 / len(testloader)
            if verbose:
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            print(
                'Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epochs, train_loss, train_acc, test_loss, test_acc,lr))
        else:
            print('Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(i+1,epochs,train_loss,train_acc,lr))

        # ====================== 使用 tensorboard ==================
        if writer is not None:
            writer.add_scalars('Loss', {'train': train_loss,
                                    'test': test_loss}, i+1)
            writer.add_scalars('Acc', {'train': train_acc ,
                                   'test': test_acc}, i+1)
            writer.add_scalar('Learning Rate',lr,i+1)
        # =========================================================
        # 如果取得更好的准确率，就保存模型
        if test_acc > best_acc:
            torch.save(net,path)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Lr = lr_list
    return Acc, Loss, Lr
```



## 可视化准确率、损失、学习率变化

这一部分用原来的，几乎没有什么变化，就更新了一下图片的大小

以下函数可以可视化准确率、损失、学习率随着迭代次数的变化

```python
def plot_history(epochs, Acc, Loss, lr):
    plt.rcParams['figure.figsize'] = (12.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1,epochs + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['test_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['test_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(epoch_list, lr)
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.show()
```

