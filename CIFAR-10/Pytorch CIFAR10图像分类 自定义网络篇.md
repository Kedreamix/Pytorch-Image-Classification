# Pytorch CIFAR10图像分类 自定义网络篇

[toc]

这里贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)

## 4.自定义网络

从网上查了很多关于神经网络的资料，无疑讨论最多的就是网络结构和参数设置，就随便弄了以下的神经网络

1.使用3*3的卷积核

2.使用初始化Xavier

3.使用BN层，减少Dropout使用 

4.使用带动量的SGD，或许也可以尝试Adam

5.默认使用ReLU（），或许可以尝试PReLU()

6.batch_size调整为2^n，一般去64,128

7.学习率大小为:0.1->0.01->0.001



首先我们还是得判断是否可以利用GPU，因为GPU的速度可能会比我们用CPU的速度快20-50倍左右，特别是对卷积神经网络来说，更是提升特别明显。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
#定义网络
class Mynet(nn.Module):# nn.Module是所有神经网络的基类，我们自己定义任何神经网络，都要继承nn.Module
    def __init__(self, num_classes=10):
        super(Mynet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,padding=1),
            
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,padding=1)
            )
        self.classifier = nn.Sequential(
            nn.Linear(32*9*9,2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes),
            )

 
    def forward(self, x):
        out = self.features(x) 
#         print(out.shape)
        out = out.view(out.size(0), -1)
#         print(out.shape)
        out = self.classifier(out)
#         print(out.shape)
        return out
net = Mynet().to(device)
```

```python
summary(net,(3,32,32))
```

> ```python
> ----------------------------------------------------------------
>         Layer (type)               Output Shape         Param #
> ================================================================
>             Conv2d-1           [-1, 64, 32, 32]           1,792
>        BatchNorm2d-2           [-1, 64, 32, 32]             128
>               ReLU-3           [-1, 64, 32, 32]               0
>             Conv2d-4          [-1, 128, 32, 32]          73,856
>        BatchNorm2d-5          [-1, 128, 32, 32]             256
>               ReLU-6          [-1, 128, 32, 32]               0
>          MaxPool2d-7          [-1, 128, 17, 17]               0
>             Conv2d-8           [-1, 64, 17, 17]          73,792
>        BatchNorm2d-9           [-1, 64, 17, 17]             128
>              ReLU-10           [-1, 64, 17, 17]               0
>            Conv2d-11           [-1, 32, 17, 17]          18,464
>       BatchNorm2d-12           [-1, 32, 17, 17]              64
>              ReLU-13           [-1, 32, 17, 17]               0
>         MaxPool2d-14             [-1, 32, 9, 9]               0
>            Linear-15                 [-1, 2048]       5,310,464
>              ReLU-16                 [-1, 2048]               0
>            Linear-17                   [-1, 10]          20,490
> ================================================================
> Total params: 5,499,434
> Trainable params: 5,499,434
> Non-trainable params: 0
> ----------------------------------------------------------------
> Input size (MB): 0.01
> Forward/backward pass size (MB): 5.47
> Params size (MB): 20.98
> Estimated Total Size (MB): 26.46
> ----------------------------------------------------------------
> ```

首先从我们summary可以看到，我们定义的模型的参数大概是5 millions，我们输入的是（batch，3，32，32）的张量，并且这里也能看到每一层后我们的图像输出大小的变化，最后输出10个参数，再通过softmax函数就可以得到我们每个类别的概率了。

我们也可以打印出我们的模型观察一下

```python
Mynet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
    (7): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=2592, out_features=2048, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=2048, out_features=10, bias=True)
  )
)
```

如果你的电脑有多个GPU，这段代码可以利用GPU进行并行计算，加快运算速度

```python
net =Mynet().to(device)
if device == 'cuda':
    net = nn.DataParallel(net)
    # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
    torch.backends.cudnn.benchmark = True
```



## 5. 定义损失函数和优化器

pytorch将深度学习中常用的优化方法全部封装在torch.optim之中，所有的优化方法都是继承基类optim.Optimizier
损失函数是封装在神经网络工具箱nn中的,包含很多损失函数

这里我使用的是SGD + momentum算法，并且我们损失函数定义为交叉熵函数，除此之外学习策略定义为动态更新学习率，如果5次迭代后，训练的损失并没有下降，那么我们便会更改学习率，会变为原来的0.5倍，最小降低到0.00001

如果想更加了解优化器和学习率策略的话，可以参考以下资料

- [Pytorch Note15 优化算法1 梯度下降（Gradient descent varients）](https://blog.csdn.net/weixin_45508265/article/details/117859824)
- [Pytorch Note16 优化算法2 动量法(Momentum)](https://blog.csdn.net/weixin_45508265/article/details/117874046)
- [Pytorch Note34 学习率衰减](https://blog.csdn.net/weixin_45508265/article/details/119089705)

这里决定迭代20次

```python
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5 ,patience = 5,min_lr = 0.000001) # 动态更新学习率
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
import time
epoch = 20
```

## 6. 训练

首先定义模型保存的位置

```python
import os
if not os.path.exists('./model'):
    os.makedirs('./model')
else:
    print('文件已存在')
save_path = './model/Mynet.pth'
```

我定义了一个train函数，在train函数中进行一个训练，并保存我们训练后的模型

```python
from utils import train
from utils import plot_history
Acc, Loss, Lr = train(net, trainloader, testloader, epoch, optimizer, criterion, scheduler, save_path, verbose = True)
```

> ```python
> Epoch [  1/ 20]  Train Loss:1.498549  Train Acc:45.00% Test Loss:1.384653  Test Acc:50.54%  Learning Rate:0.100000	Time 00:43
> Epoch [  2/ 20]  Train Loss:1.059985  Train Acc:62.00% Test Loss:1.016556  Test Acc:63.80%  Learning Rate:0.100000	Time 00:40
> Epoch [  3/ 20]  Train Loss:0.874394  Train Acc:68.95% Test Loss:0.899891  Test Acc:68.67%  Learning Rate:0.100000	Time 00:42
> Epoch [  4/ 20]  Train Loss:0.777563  Train Acc:72.65% Test Loss:0.867772  Test Acc:69.60%  Learning Rate:0.100000	Time 00:44
> Epoch [  5/ 20]  Train Loss:0.699190  Train Acc:75.54% Test Loss:0.812787  Test Acc:71.54%  Learning Rate:0.100000	Time 00:42
> Epoch [  6/ 20]  Train Loss:0.657028  Train Acc:77.06% Test Loss:0.847193  Test Acc:70.65%  Learning Rate:0.100000	Time 00:42
> Epoch [  7/ 20]  Train Loss:0.625934  Train Acc:78.05% Test Loss:0.714590  Test Acc:75.08%  Learning Rate:0.100000	Time 00:43
> Epoch [  8/ 20]  Train Loss:0.594711  Train Acc:79.31% Test Loss:0.989479  Test Acc:68.00%  Learning Rate:0.100000	Time 00:42
> Epoch [  9/ 20]  Train Loss:0.576213  Train Acc:79.96% Test Loss:0.836162  Test Acc:72.17%  Learning Rate:0.100000	Time 00:41
> Epoch [ 10/ 20]  Train Loss:0.559027  Train Acc:80.56% Test Loss:0.713146  Test Acc:75.34%  Learning Rate:0.100000	Time 00:41
> Epoch [ 11/ 20]  Train Loss:0.535767  Train Acc:81.35% Test Loss:0.774732  Test Acc:75.33%  Learning Rate:0.100000	Time 00:39
> Epoch [ 12/ 20]  Train Loss:0.521346  Train Acc:81.88% Test Loss:0.624320  Test Acc:79.46%  Learning Rate:0.100000	Time 00:40
> Epoch [ 13/ 20]  Train Loss:0.504253  Train Acc:82.64% Test Loss:0.855251  Test Acc:71.86%  Learning Rate:0.100000	Time 00:40
> Epoch [ 14/ 20]  Train Loss:0.499133  Train Acc:82.75% Test Loss:0.677991  Test Acc:76.81%  Learning Rate:0.100000	Time 00:39
> Epoch [ 15/ 20]  Train Loss:0.483084  Train Acc:83.32% Test Loss:0.642261  Test Acc:79.37%  Learning Rate:0.100000	Time 00:40
> Epoch [ 16/ 20]  Train Loss:0.476981  Train Acc:83.51% Test Loss:0.731425  Test Acc:76.28%  Learning Rate:0.100000	Time 00:40
> Epoch [ 17/ 20]  Train Loss:0.477170  Train Acc:83.54% Test Loss:0.584530  Test Acc:80.19%  Learning Rate:0.100000	Time 00:40
> Epoch [ 18/ 20]  Train Loss:0.459746  Train Acc:83.99% Test Loss:0.664121  Test Acc:78.35%  Learning Rate:0.100000	Time 00:38
> Epoch [ 19/ 20]  Train Loss:0.449382  Train Acc:84.55% Test Loss:0.579332  Test Acc:79.91%  Learning Rate:0.100000	Time 00:38
> Epoch [ 20/ 20]  Train Loss:0.451750  Train Acc:84.34% Test Loss:0.608241  Test Acc:79.69%  Learning Rate:0.100000	Time 00:40
> ```

接着可以分别打印，损失函数曲线，准确率曲线和学习率曲线

```python
plot_history(epoch ,Acc, Loss, Lr)
```

### 损失函数曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/f094f064403a47f196431242bb50e27b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

### 准确率曲线



![在这里插入图片描述](https://img-blog.csdnimg.cn/6ff73a0bc69b4eb7ac7f645f73ed41e2.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

### 学习率曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/f738c2508f3f43d4819e9e2e4aaac8db.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 7.测试

### 查看准确率

```python
correct = 0   # 定义预测正确的图片数，初始化为0
total = 0     # 总共参与测试的图片数，也初始化为0
# testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
for data in testloader:  # 循环每一个batch
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    net.eval()  # 把模型转为test模式
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    outputs = net(images)  # 输入网络进行测试
    
    # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)          # 更新测试图片的数量
    correct += (predicted == labels).sum() # 更新正确分类的图片的数量

print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
 
```

> ```python
> Accuracy of the network on the 10000 test images: 79.43 %
> ```

可以看到自定义网络的模型在测试集中准确率达到79.43%



程序中的 `torch.max(outputs.data, 1)` ，返回一个tuple (元组)

而这里很明显，这个返回的元组的第一个元素是image data，即是最大的 值，第二个元素是label， 即是最大的值 的 索引！我们只需要label（最大值的索引），所以就会有` _ `,predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给` _`， 就是舍弃它的意思；

### 查看每一类的准确率

```python
 # 定义2个存储每类中测试正确的个数的 列表，初始化为0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
# testloader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True, num_workers=2)
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
    #4组(batch_size)数据中，输出于label相同的，标记为1，否则为0
        c = (predicted == labels).squeeze()
        for i in range(len(images)):      # 因为每个batch都有4张图片，所以还需要一个4的小循环
            label = labels[i]   # 对各个类的进行各自累加
            class_correct[label] += c[i]
            class_total[label] += 1
 
 
for i in range(10):
    print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
```

> ```python
> Accuracy of airplane : 89.00 %
> Accuracy of automobile : 94.00 %
> Accuracy of  bird : 73.20 %
> Accuracy of   cat : 61.30 %
> Accuracy of  deer : 68.10 %
> Accuracy of   dog : 75.80 %
> Accuracy of  frog : 84.40 %
> Accuracy of horse : 74.80 %
> Accuracy of  ship : 88.90 %
> Accuracy of truck : 84.00 %
> ```

### 抽样测试并可视化一部分结果

```python
dataiter = iter(testloader)
images, labels = dataiter.next()
images_ = images
#images_ = images_.view(images.shape[0], -1)
images_ = images_.to(device)
labels = labels.to(device)
val_output = net(images_)
_, val_preds = torch.max(val_output, 1)

fig = plt.figure(figsize=(25,4))

correct = torch.sum(val_preds == labels.data).item()

val_preds = val_preds.cpu()
labels = labels.cpu()

print("Accuracy Rate = {}%".format(correct/len(images) * 100))

fig = plt.figure(figsize=(25,25))
for idx in np.arange(64):    
    ax = fig.add_subplot(8, 8, idx+1, xticks=[], yticks=[])
    #fig.tight_layout()
#     plt.imshow(im_convert(images[idx]))
    imshow(images[idx])
    ax.set_title("{}, ({})".format(classes[val_preds[idx].item()], classes[labels[idx].item()]), 
                 color = ("green" if val_preds[idx].item()==labels[idx].item() else "red"))
```

> ```python
> Accuracy Rate = 80.859375%
> <Figure size 1800x288 with 0 Axes>
> ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/ac4f09f06ef84902bbeaece032173b2c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 8. 保存模型

```python
torch.save(net,save_path[:-4]+'_'+str(epoch)+'.pth')
# torch.save(net, './model/MyNet.pth')
```

## 9. 预测

### 读取本地图片进行预测

```python
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Mynet()

model = torch.load(save_path)  # 加载模型
# model = model.to('cuda')
model.eval()  # 把模型转为test模式

# 读取要预测的图片
img = Image.open("./airplane.jpg").convert('RGB') # 读取图像
```

```
img
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/a4baf7e595b54cc1b3221e57b62ec43d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

接着我们就进行预测图片，不过这里有一个点，我们需要对我们的图片也进行transforms，因为我们的训练的时候，对每个图像也是进行了transforms的，所以我们需要保持一致

```python
trans = transforms.Compose([transforms.Scale((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                 std=(0.5, 0.5, 0.5)),
                           ])
 
img = trans(img)
img = img.to(device)
# 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
img = img.unsqueeze(0)  
    # 扩展后，为[1，3，32，32]
output = model(img)
prob = F.softmax(output,dim=1) #prob是10个分类的概率
print("概率",prob)
value, predicted = torch.max(output.data, 1)
print("类别",predicted.item())
print(value)
pred_class = classes[predicted.item()]
print("分类",pred_class)
```

> ```python
> 概率 tensor([[9.9798e-01, 4.7527e-05, 1.3889e-04, 1.8324e-05, 5.1395e-06, 5.8237e-07,
>          8.4135e-07, 3.9314e-07, 1.7947e-03, 1.3177e-05]], device='cuda:0',
>        grad_fn=<SoftmaxBackward>)
> 类别 0
> tensor([10.2566], device='cuda:0')
> 分类 plane
> ```

这里就可以看到，我们最后的结果，分类为plane，我们的置信率大概是99.79%

### 读取图片地址进行预测

我们也可以通过读取图片的url地址进行预测，这里我找了多个不同的图片进行预测

```python
import requests
from PIL import Image
url = 'https://dss2.bdstatic.com/70cFvnSh_Q1YnxGkpoWK1HF6hhy/it/u=947072664,3925280208&fm=26&gp=0.jpg'
url = 'https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=2952045457,215279295&fm=26&gp=0.jpg'
url = 'https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=2838383012,1815030248&fm=26&gp=0.jpg'
url = 'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.goupuzi.com%2Fnewatt%2FMon_1809%2F1_179223_7463b117c8a2c76.jpg&refer=http%3A%2F%2Fwww.goupuzi.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1624346733&t=36ba18326a1e010737f530976201326d'
url = 'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=2799543344,3604342295&fm=224&gp=0.jpg'
# url = 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=2032505694,2851387785&fm=26&gp=0.jpg'
response = requests.get(url, stream=True)
print (response)
img = Image.open(response.raw)
img
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/c22f662ef70c4539bad1ad21ee8b702d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

这里和前面是一样的

```python
trans = transforms.Compose([transforms.Scale((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                 std=(0.5, 0.5, 0.5)),
                           ])
 
img = trans(img)
img = img.to(device)
# 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
img = img.unsqueeze(0)  
    # 扩展后，为[1，3，32，32]
output = model(img)
prob = F.softmax(output,dim=1) #prob是10个分类的概率
print("概率",prob)
value, predicted = torch.max(output.data, 1)
print("类别",predicted.item())
print(value)
pred_class = classes[predicted.item()]
print("分类",pred_class)
```

> ```python
> 概率 tensor([[1.4595e-01, 1.2156e-04, 8.0566e-01, 1.4928e-02, 7.2242e-03, 9.0186e-03,
>          4.7708e-03, 9.8267e-04, 1.0993e-02, 3.5262e-04]], device='cuda:0',
>        grad_fn=<SoftmaxBackward>)
> 类别 2
> tensor([4.7554], device='cuda:0')
> 分类 bird
> ```

可以看到，我们分类的结果是bird，而真正的分类是cat，所以这个模型还不是特别完美，还需要不断完善



## 10.总结

通过这次Pytorch CIFAR10图像分类 自定义网络篇，我们学会了如何构造一个网络模型，并且在一个较大的数据集也能取得比较好的结果，接下来会用深度学习的开山之作LeNet作为我们的网络模型测试一下CIFAR10，看看会有什么比较好的效果。



顺带提一句，我们的数据和代码都在我的汇总篇里有说明，如果需要，可以自取

这里再贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)