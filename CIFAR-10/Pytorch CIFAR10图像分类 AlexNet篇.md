# Pytorch CIFAR10图像分类 AlexNet篇

[toc]

## 4.定义网络（AlexNet）

![在这里插入图片描述](img\AlexNet.png)

**AlexNet 结构：**

- 输入层：图像大小为 227×227×3，其中 3 表示输入图像的 channel 数（R，G，B）为 3。

- 卷积层：filter 大小 11×11，filter 个数 96，卷积步长 s=4。（filter 大小只列出了宽和高，filter矩阵的 channel 数和输入图片的 channel 数一样，在这里没有列出）

- 池化层：max pooling，filter 大小 3×3，步长 s=2。

- 卷积层：filter 大小 5×5，filter 个数 256，步长 s=1，padding 使用 same convolution，即使得卷积层输出图像和输入图像在宽和高上保持不变。

- 池化层：max pooling，filter 大小 3×3，步长 s=2。

- 卷积层：filter 大小 3×3，filter 个数 384，步长 s=1，padding 使用 same convolution。

- 卷积层：filter 大小 3×3，filter 个数 384，步长 s=1，padding 使用 same convolution。

- 卷积层：filter 大小 3×3，filter 个数 256，步长 s=1，padding 使用 same convolution。

- 池化层：max pooling，filter 大小 3×3，步长 s=2；池化操作结束后，将大小为  6×6×256 的输出矩阵 flatten 成一个 9216 维的向量。

- 全连接层：neuron 数量为 4096。

- 全连接层：neuron 数量为 4096。

- 全连接层，输出层：softmax 激活函数，neuron 数量为 1000，代表 1000 个类别。

**AlexNet 一些性质：**

- 大约 60million 个参数；
- 使用 ReLU 作为激活函数。



首先我们还是得判断是否可以利用GPU，因为GPU的速度可能会比我们用CPU的速度快20-50倍左右，特别是对卷积神经网络来说，更是提升特别明显。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
# 定义神经网络  
class AlexNet(nn.Module):  # 训练 ALexNet
    '''
    5层卷积，3层全连接
    ''' 
    def __init__(self):
        super(AlexNet, self).__init__()
        # 五个卷积层 输入 32 * 32 * 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),   # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )                            # 最后一层卷积层，输出 1 * 1 * 128
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size()[0],-1)
        x = self.dense(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    

net = AlexNet().to(device)
```

```python
summary(net,(3,32,32))
```

> ```python
> ----------------------------------------------------------------
>         Layer (type)               Output Shape         Param #
> ================================================================
>             Conv2d-1            [-1, 6, 32, 32]             168
>               ReLU-2            [-1, 6, 32, 32]               0
>          MaxPool2d-3            [-1, 6, 16, 16]               0
>             Conv2d-4           [-1, 16, 16, 16]             880
>               ReLU-5           [-1, 16, 16, 16]               0
>          MaxPool2d-6             [-1, 16, 8, 8]               0
>             Conv2d-7             [-1, 32, 8, 8]           4,640
>               ReLU-8             [-1, 32, 8, 8]               0
>          MaxPool2d-9             [-1, 32, 4, 4]               0
>            Conv2d-10             [-1, 64, 4, 4]          18,496
>              ReLU-11             [-1, 64, 4, 4]               0
>         MaxPool2d-12             [-1, 64, 2, 2]               0
>            Conv2d-13            [-1, 128, 2, 2]          73,856
>              ReLU-14            [-1, 128, 2, 2]               0
>         MaxPool2d-15            [-1, 128, 1, 1]               0
>            Linear-16                  [-1, 120]          15,480
>              ReLU-17                  [-1, 120]               0
>            Linear-18                   [-1, 84]          10,164
>              ReLU-19                   [-1, 84]               0
>            Linear-20                   [-1, 10]             850
> ================================================================
> Total params: 124,534
> Trainable params: 124,534
> Non-trainable params: 0
> ----------------------------------------------------------------
> Input size (MB): 0.01
> Forward/backward pass size (MB): 0.24
> Params size (MB): 0.48
> Estimated Total Size (MB): 0.73
> ----------------------------------------------------------------
> ```

首先从我们summary可以看到，我们定义的模型的参数大概是124 thousands，我们输入的是（batch，3，32，32）的张量，并且这里也能看到每一层后我们的图像输出大小的变化，最后输出10个参数，再通过softmax函数就可以得到我们每个类别的概率了。

我们也可以打印出我们的模型观察一下

```python
AlexNet(
  (conv1): Sequential(
    (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv4): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv5): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense): Sequential(
    (0): Linear(in_features=128, out_features=120, bias=True)
    (1): ReLU()
    (2): Linear(in_features=120, out_features=84, bias=True)
    (3): ReLU()
    (4): Linear(in_features=84, out_features=10, bias=True)
  )
)
```

如果你的电脑有多个GPU，这段代码可以利用GPU进行并行计算，加快运算速度

```python
net =AlexNet().to(device)
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
save_path = './model/AlexNet.pth'
```

我定义了一个train函数，在train函数中进行一个训练，并保存我们训练后的模型

```python
from utils import trainfrom utils import plot_historyAcc, Loss, Lr = train(net, trainloader, testloader, epoch, optimizer, criterion, scheduler, save_path, verbose = True)
```

> ```python
> Epoch [  1/ 20]  Train Loss:2.264505  Train Acc:12.24% Test Loss:2.108758  Test Acc:19.67%  Learning Rate:0.100000	Time 00:17
> Epoch [  2/ 20]  Train Loss:1.946763  Train Acc:25.11% Test Loss:1.768384  Test Acc:29.92%  Learning Rate:0.100000	Time 00:16
> Epoch [  3/ 20]  Train Loss:1.731969  Train Acc:34.11% Test Loss:1.638685  Test Acc:39.35%  Learning Rate:0.100000	Time 00:15
> Epoch [  4/ 20]  Train Loss:1.579956  Train Acc:40.75% Test Loss:1.582188  Test Acc:42.75%  Learning Rate:0.100000	Time 00:17
> Epoch [  5/ 20]  Train Loss:1.468241  Train Acc:46.01% Test Loss:1.425271  Test Acc:47.40%  Learning Rate:0.100000	Time 00:16
> Epoch [  6/ 20]  Train Loss:1.376591  Train Acc:50.15% Test Loss:1.440676  Test Acc:48.79%  Learning Rate:0.100000	Time 00:17
> Epoch [  7/ 20]  Train Loss:1.294923  Train Acc:53.45% Test Loss:1.309682  Test Acc:52.91%  Learning Rate:0.100000	Time 00:16
> Epoch [  8/ 20]  Train Loss:1.256029  Train Acc:55.13% Test Loss:1.367346  Test Acc:50.88%  Learning Rate:0.100000	Time 00:16
> Epoch [  9/ 20]  Train Loss:1.226186  Train Acc:56.52% Test Loss:1.170674  Test Acc:58.46%  Learning Rate:0.100000	Time 00:15
> Epoch [ 10/ 20]  Train Loss:1.183964  Train Acc:58.39% Test Loss:1.271486  Test Acc:55.51%  Learning Rate:0.100000	Time 00:16
> Epoch [ 11/ 20]  Train Loss:1.166563  Train Acc:59.03% Test Loss:1.178845  Test Acc:58.78%  Learning Rate:0.100000	Time 00:16
> Epoch [ 12/ 20]  Train Loss:1.126194  Train Acc:60.49% Test Loss:1.151750  Test Acc:59.32%  Learning Rate:0.100000	Time 00:15
> Epoch [ 13/ 20]  Train Loss:1.108992  Train Acc:61.17% Test Loss:1.114536  Test Acc:61.64%  Learning Rate:0.100000	Time 00:18
> Epoch [ 14/ 20]  Train Loss:1.101274  Train Acc:61.50% Test Loss:1.130676  Test Acc:60.41%  Learning Rate:0.100000	Time 00:17
> Epoch [ 15/ 20]  Train Loss:1.092924  Train Acc:62.09% Test Loss:1.105640  Test Acc:61.33%  Learning Rate:0.100000	Time 00:16
> Epoch [ 16/ 20]  Train Loss:1.077327  Train Acc:62.56% Test Loss:1.128582  Test Acc:60.12%  Learning Rate:0.100000	Time 00:16
> Epoch [ 17/ 20]  Train Loss:1.056912  Train Acc:63.45% Test Loss:1.043383  Test Acc:63.83%  Learning Rate:0.100000	Time 00:16
> Epoch [ 18/ 20]  Train Loss:1.051743  Train Acc:63.53% Test Loss:1.060718  Test Acc:63.85%  Learning Rate:0.100000	Time 00:16
> Epoch [ 19/ 20]  Train Loss:1.052119  Train Acc:63.61% Test Loss:1.047150  Test Acc:62.99%  Learning Rate:0.100000	Time 00:15
> Epoch [ 20/ 20]  Train Loss:1.032711  Train Acc:64.17% Test Loss:1.087847  Test Acc:61.46%  Learning Rate:0.100000	Time 00:15
> ```

接着可以分别打印，损失函数曲线，准确率曲线和学习率曲线

```python
plot_history(epoch ,Acc, Loss, Lr)
```

### 损失函数曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/716a42c51a7140bbbf8c17ba02505957.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

### 准确率曲线



![在这里插入图片描述](https://img-blog.csdnimg.cn/78949732c61f4031926ff4ec2e59b70a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

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
> Accuracy of the network on the 10000 test images: 60.80 %
> ```

可以看到自定义网络的模型在测试集中准确率达到60.80%



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
> Accuracy of airplane : 56.40 %
> Accuracy of automobile : 66.70 %
> Accuracy of  bird : 27.50 %
> Accuracy of   cat : 76.10 %
> Accuracy of  deer : 62.10 %
> Accuracy of   dog : 29.10 %
> Accuracy of  frog : 68.30 %
> Accuracy of horse : 65.70 %
> Accuracy of  ship : 78.30 %
> Accuracy of truck : 78.40 %
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
> Accuracy Rate = 54.6875%
> <Figure size 1800x288 with 0 Axes>
> ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/7fcc510694ed42a1b9fc9a603952c52f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

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
> 概率 tensor([[8.1557e-01, 3.5077e-03, 5.7083e-03, 1.5977e-04, 4.3912e-03, 9.9246e-06,
>          1.6540e-04, 2.8073e-05, 1.6927e-01, 1.1888e-03]], device='cuda:0',
>        grad_fn=<SoftmaxBackward>)
> 类别 0
> tensor([6.2487], device='cuda:0')
> 分类 plane
> ```

这里就可以看到，我们最后的结果，分类为plane，我们的置信率大概是81.57%，很明显，这个分类是有问题的。

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
> 概率 tensor([[0.8136, 0.0008, 0.1219, 0.0113, 0.0282, 0.0047, 0.0074, 0.0022, 0.0081,
>          0.0018]], device='cuda:0', grad_fn=<SoftmaxBackward>)
> 类别 0
> tensor([4.2904], device='cuda:0')
> 分类 plane
> ```

可以看到，我们分类的结果是plane，而真正的分类是cat，所以这个模型还不是特别完美，还需要不断完善



## 10.总结

在这个模型中，其实我们只迭代了20次，AlexNet曾经在ILSRC2012上取得了冠军，但是可能我们的训练次数太少了，所以使得模型还未收敛，如果想提高AlexNet模型的准确率，可以对他进行加深迭代次数，我相信可以取得比较好的结果。



顺带提一句，我们的数据和代码都在我的汇总篇里有说明，如果需要，可以自取

这里再贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)



