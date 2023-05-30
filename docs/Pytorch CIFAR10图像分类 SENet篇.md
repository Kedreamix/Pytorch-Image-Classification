# Pytorch CIFAR10图像分类 SENet篇

[toc]

再次介绍一下我的专栏，很适合大家初入深度学习或者是Pytorch和Keras，希望这能够帮助初学深度学习的同学一个入门Pytorch或者Keras的项目和在这之中更加了解Pytorch&Keras和各个图像分类的模型。

他有比较清晰的可视化结构和架构，除此之外，我是用jupyter写的，所以说在文章整体架构可以说是非常清晰，可以帮助你快速学习到各个模块的知识，而不是通过python脚本一行一行的看，这样的方式是符合初学者的。

除此之外，如果你需要变成脚本形式，也是很简单的。
这里贴一下汇总篇：[汇总篇](https://redamancy.blog.csdn.net/article/details/127859300)

## 4. 定义网络（SENet）

我们之前已经学习了 AlexNet，VGGNet，InceptionNet，ResNet，DenseNet等，他们的效果已经被充分验证，而且被广泛的应用在各类计算机视觉任务上。这里我们再学习一个网络（SENet）。Squeeze-and-Excitation Networks（SENet）是由自动驾驶公司Momenta在2017年公布的一种全新的图像识别结构，它通过对特征通道间的相关性进行建模，把重要的特征进行强化来提升准确率。SENet 以极大的优势获得了最后一届 ImageNet 2017 竞赛 Image Classification 任务的冠军，top5的错误率达到了2.251%，比2016年的第一名还要低25%，可谓提升巨大。下面就来具体学习一下SENet。



### SE Block

SE Block是SENet的Block单元，图中的Ftr是传统的卷积结构，X和U是Ftr的输入（C'xH'xW'）和输出（CxHxW），这些都是以往结构中已存在的。SENet增加的部分是U后的结构：对U先做一个Global Average Pooling（图中的Fsq(.)，作者称为Squeeze过程），输出的1x1xC数据再经过两级全连接（图中的Fex(.)，作者称为Excitation过程），最后用sigmoid（论文中的self-gating mechanism）限制到[0，1]的范围，把这个值作为scale乘到U的C个通道上， 作为下一级的输入数据。这种结构的原理是想通过控制scale的大小，把重要的特征增强，不重要的特征减弱，从而让提取的特征指向性更强。

![img](https://img-blog.csdn.net/20180423230918755)

上图是SE 模块的示意图。给定一个输入 x，其特征通道数为 c_1，通过一系列卷积等一般变换后得到一个特征通道数为 c_2 的特征。与传统的 CNN 不一样的是，接下来通过三个操作来重标定前面得到的特征。

首先是 Squeeze 操作，顺着空间维度来进行特征压缩，将每个二维的特征通道变成一个实数，这个实数某种程度上具有全局的感受野，并且输出的维度和输入的特征通道数相匹配。它表征着在特征通道上响应的全局分布，而且使得靠近输入的层也可以获得全局的感受野，这一点在很多任务中都是非常有用的。

其次是 Excitation 操作，它是一个类似于循环神经网络中门的机制。通过参数 w 来为每个特征通道生成权重，其中参数 w 被学习用来显式地建模特征通道间的相关性。

最后是一个 Reweight 的操作，将 Excitation 的输出的权重看做是进过特征选择后的每个特征通道的重要性，然后通过乘法逐通道加权到先前的特征上，完成在通道维度上的对原始特征的重标定。

我们可以总结一下**中心思想**：对于每个输出 channel，预测一个常数权重，对每个 channel 加权一下，本质上，SE模块是在 channel 维度上做 attention 或者 gating 操作，这种注意力机制让模型可以更加关注信息量最大的 channel 特征，而抑制那些不重要的 channel 特征。SENet 一个很大的优点就是可以很方便地集成到现有网络中，提升网络性能，并且代价很小。

### SE模块的应用

这是两个SENet实际应用的例子，左侧是SE-Inception的结构，即Inception模块和SENet组和在一起；右侧是SE-ResNet，ResNet和SENet的组合，这种结构scale放到了直连相加之前。

![img](https://img-blog.csdn.net/20180423233511251)



SE模块可以嵌入到现在几乎所有的网络结构中。通过在原始网络结构的 building block 单元中嵌入 SE模块，我们可以获得不同种类的 SENet。如SE-BN-Inception，SE-ResNet，SE-ReNeXt，SE-Inception-ResNet-v2等等

![img](https://img2020.cnblogs.com/blog/1226410/202101/1226410-20210122143308379-1799927935.png)

从上面的介绍中可以发现，SENet构造非常简单，而且很容易被部署，不需要引入新的函数或者层。除此之外，它还在模型和计算复杂度上具有良好的特性。拿 ResNet-50 和 SE-ResNet-50 对比举例来说，SE-ResNet-50 相对于 ResNet-50有着 10% 模型参数的增长。额外的模型参数都存在于 Bottleneck 设计的两个 Fully Connected 中，由于 ResNet 结构中最后一个 stage 的特征通道数目为 2048，导致模型参数有着较大的增长，实现发现移除掉最后一个 stage 中 3个 build block 上的 SE设定，可以将 10%参数量的增长减少到 2%。此时模型的精度几乎无损失。

![img](https://img2020.cnblogs.com/blog/1226410/202101/1226410-20210122143339705-1444052988.png)

首先我们还是得判断是否可以利用GPU，因为GPU的速度可能会比我们用CPU的速度快20-50倍左右，特别是对卷积神经网络来说，更是提升特别明显。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

接下来我们来构建SE-ResNet方法，首先第一步我们还是要构建一个BasicBlock，这里也加入了SEBlock

```python

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        """
        构造函数，初始化BasicBlock模块的各个组件

        :param in_planes: 输入通道数
        :param planes: 输出通道数
        :param stride: 步长
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # 如果输入输出通道数不同或者步长不为1，则使用1x1卷积层
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w 

        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

在激活前，再定义一下一个PreActBlock

```python

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out

```

定义了基本模块后，我们就构建SENet的结构

```python

class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SENet18(num_classes=10):
    return SENet(PreActBlock, [2,2,2,2], num_classes=num_classes)


```

```python
net = SENet18(num_classes=10).to(device)
```

### summary查看网络

我们可以通过summary来看到，模型的维度的变化，这个也是和论文是匹配的，经过层后shape的变化，是否最后也是输出(batch,shape)

```python
summary(net,(2,3,32,32))
```



> ```python
> ----------------------------------------------------------------
>         Layer (type)               Output Shape         Param #
> ================================================================
>             Conv2d-1           [-1, 64, 32, 32]           1,728
>        BatchNorm2d-2           [-1, 64, 32, 32]             128
>        BatchNorm2d-3           [-1, 64, 32, 32]             128
>             Conv2d-4           [-1, 64, 32, 32]          36,864
>        BatchNorm2d-5           [-1, 64, 32, 32]             128
>             Conv2d-6           [-1, 64, 32, 32]          36,864
>             Conv2d-7              [-1, 4, 1, 1]             260
>             Conv2d-8             [-1, 64, 1, 1]             320
>            SEBlock-9           [-1, 64, 32, 32]               0
>       BatchNorm2d-10           [-1, 64, 32, 32]             128
>            Conv2d-11           [-1, 64, 32, 32]          36,864
>       BatchNorm2d-12           [-1, 64, 32, 32]             128
>            Conv2d-13           [-1, 64, 32, 32]          36,864
>            Conv2d-14              [-1, 4, 1, 1]             260
>            Conv2d-15             [-1, 64, 1, 1]             320
>           SEBlock-16           [-1, 64, 32, 32]               0
>       BatchNorm2d-17           [-1, 64, 32, 32]             128
>            Conv2d-18          [-1, 128, 16, 16]           8,192
>            Conv2d-19          [-1, 128, 16, 16]          73,728
>       BatchNorm2d-20          [-1, 128, 16, 16]             256
>            Conv2d-21          [-1, 128, 16, 16]         147,456
>            Conv2d-22              [-1, 8, 1, 1]           1,032
>            Conv2d-23            [-1, 128, 1, 1]           1,152
>           SEBlock-24          [-1, 128, 16, 16]               0
>       BatchNorm2d-25          [-1, 128, 16, 16]             256
>            Conv2d-26          [-1, 128, 16, 16]         147,456
>       BatchNorm2d-27          [-1, 128, 16, 16]             256
>            Conv2d-28          [-1, 128, 16, 16]         147,456
>            Conv2d-29              [-1, 8, 1, 1]           1,032
>            Conv2d-30            [-1, 128, 1, 1]           1,152
>           SEBlock-31          [-1, 128, 16, 16]               0
>       BatchNorm2d-32          [-1, 128, 16, 16]             256
>            Conv2d-33            [-1, 256, 8, 8]          32,768
>            Conv2d-34            [-1, 256, 8, 8]         294,912
>       BatchNorm2d-35            [-1, 256, 8, 8]             512
>            Conv2d-36            [-1, 256, 8, 8]         589,824
>            Conv2d-37             [-1, 16, 1, 1]           4,112
>            Conv2d-38            [-1, 256, 1, 1]           4,352
>           SEBlock-39            [-1, 256, 8, 8]               0
>       BatchNorm2d-40            [-1, 256, 8, 8]             512
>            Conv2d-41            [-1, 256, 8, 8]         589,824
>       BatchNorm2d-42            [-1, 256, 8, 8]             512
>            Conv2d-43            [-1, 256, 8, 8]         589,824
>            Conv2d-44             [-1, 16, 1, 1]           4,112
>            Conv2d-45            [-1, 256, 1, 1]           4,352
>           SEBlock-46            [-1, 256, 8, 8]               0
>       BatchNorm2d-47            [-1, 256, 8, 8]             512
>            Conv2d-48            [-1, 512, 4, 4]         131,072
>            Conv2d-49            [-1, 512, 4, 4]       1,179,648
>       BatchNorm2d-50            [-1, 512, 4, 4]           1,024
>            Conv2d-51            [-1, 512, 4, 4]       2,359,296
>            Conv2d-52             [-1, 32, 1, 1]          16,416
>            Conv2d-53            [-1, 512, 1, 1]          16,896
>           SEBlock-54            [-1, 512, 4, 4]               0
>       BatchNorm2d-55            [-1, 512, 4, 4]           1,024
>            Conv2d-56            [-1, 512, 4, 4]       2,359,296
>       BatchNorm2d-57            [-1, 512, 4, 4]           1,024
>            Conv2d-58            [-1, 512, 4, 4]       2,359,296
>            Conv2d-59             [-1, 32, 1, 1]          16,416
>            Conv2d-60            [-1, 512, 1, 1]          16,896
>           SEBlock-61            [-1, 512, 4, 4]               0
>            Linear-62                   [-1, 10]           5,130
> ================================================================
> Total params: 11,260,354
> Trainable params: 11,260,354
> Non-trainable params: 0
> ----------------------------------------------------------------
> Input size (MB): 0.01
> Forward/backward pass size (MB): 11.27
> Params size (MB): 42.95
> Estimated Total Size (MB): 54.23
> ----------------------------------------------------------------
> 
> ```

首先从我们summary可以看到，我们输入的是（batch，3，32，32）的张量，并且这里也能看到每一层后我们的图像输出大小的变化，最后输出10个参数，再通过softmax函数就可以得到我们每个类别的概率了。

我们也可以打印出我们的模型观察一下

```python
SENet(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1): Sequential(
    (0): SEBlock(
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (fc1): Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(4, 64, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SEBlock(
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (fc1): Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(4, 64, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (layer2): Sequential(
    (0): SEBlock(
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (shortcut): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      )
      (fc1): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(8, 128, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SEBlock(
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (fc1): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(8, 128, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (layer3): Sequential(
    (0): SEBlock(
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (shortcut): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      )
      (fc1): Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SEBlock(
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (fc1): Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (layer4): Sequential(
    (0): SEBlock(
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (shortcut): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
      )
      (fc1): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SEBlock(
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (fc1): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
      (fc2): Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (linear): Linear(in_features=512, out_features=10, bias=True)
)

```

### 测试和定义网络

接下来可以简单测试一下，是否输入后能得到我们的正确的维度shape

```python
test_x = torch.randn(2,3,32,32).to(device)
test_y = net(test_x)
print(test_y.shape)
```

```bash
torch.Size([2, 10])
```

定义网络和设置类别

```python
net = SENet(num_classes=10)
```



## 5. 定义损失函数和优化器

pytorch将深度学习中常用的优化方法全部封装在torch.optim之中，所有的优化方法都是继承基类optim.Optimizier
损失函数是封装在神经网络工具箱nn中的,包含很多损失函数

这里我使用的是SGD + momentum算法，并且我们损失函数定义为交叉熵函数，除此之外学习策略定义为动态更新学习率，如果5次迭代后，训练的损失并没有下降，那么我们便会更改学习率，会变为原来的0.5倍，最小降低到0.00001

如果想更加了解优化器和学习率策略的话，可以参考以下资料

- [Pytorch Note15 优化算法1 梯度下降（Gradient descent varients）](https://blog.csdn.net/weixin_45508265/article/details/117859824)
- [Pytorch Note16 优化算法2 动量法(Momentum)](https://blog.csdn.net/weixin_45508265/article/details/117874046)
- [Pytorch Note34 学习率衰减](https://blog.csdn.net/weixin_45508265/article/details/119089705)

这里决定迭代10次

```python
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94 ,patience = 1,min_lr = 0.000001) # 动态更新学习率
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
import time
epoch = 10
```

## 6. 训练及可视化（增加TensorBoard可视化）

首先定义模型保存的位置

```python
import os
if not os.path.exists('./model'):
    os.makedirs('./model')
else:
    print('文件已存在')
save_path = './model/SENet.pth'
```

这次更新了tensorboard的可视化，可以得到更好看的图片，并且能可视化出不错的结果

```python
# 使用tensorboard
from torch.utils.tensorboard import SummaryWriter
os.makedirs("./logs", exist_ok=True)
tbwriter = SummaryWriter(log_dir='./logs/SENet', comment='SENet')  # 使用tensorboard记录中间输出
tbwriter.add_graph(model= net, input_to_model=torch.randn(size=(1, 3, 32, 32)))
```

如果存在GPU可以选择使用GPU进行运行，并且可以设置并行运算

```python
if device == 'cuda':
    net.to(device)
    net = nn.DataParallel(net) # 使用并行运算
```

### 开始训练

我定义了一个train函数，在train函数中进行一个训练，并保存我们训练后的模型，这一部分一定要注意，这里的utils文件是我个人写的，所以需要下载下来

或者可以参考我们的[工具函数篇](https://redamancy.blog.csdn.net/article/details/127856569)，我还更新了结果和方法，利用tqdm更能可视化我们的结果。

```python
from utils import plot_history
from utils import train
Acc, Loss, Lr = train(net, trainloader, testloader, epoch, optimizer, criterion, scheduler, save_path, tbwriter, verbose = True)
```

> ```bash
> Train Epoch 1/20: 100%|██████████| 390/390 [00:13<00:00, 28.28it/s, Train Acc=0.47, Train Loss=1.44] 
> Test Epoch 1/20: 100%|██████████| 78/78 [00:01<00:00, 64.03it/s, Test Acc=0.546, Test Loss=1.28]
> Epoch [  1/ 20]  Train Loss:1.444286  Train Acc:46.98% Test Loss:1.276667  Test Acc:54.61%  Learning Rate:0.100000
> Train Epoch 2/20: 100%|██████████| 390/390 [00:13<00:00, 28.94it/s, Train Acc=0.686, Train Loss=0.879]
> Test Epoch 2/20: 100%|██████████| 78/78 [00:01<00:00, 60.01it/s, Test Acc=0.663, Test Loss=0.986]
> Epoch [  2/ 20]  Train Loss:0.878925  Train Acc:68.59% Test Loss:0.986204  Test Acc:66.26%  Learning Rate:0.100000
> Train Epoch 3/20: 100%|██████████| 390/390 [00:13<00:00, 28.15it/s, Train Acc=0.772, Train Loss=0.658]
> Test Epoch 3/20: 100%|██████████| 78/78 [00:01<00:00, 63.16it/s, Test Acc=0.771, Test Loss=0.649]
> Epoch [  3/ 20]  Train Loss:0.657689  Train Acc:77.20% Test Loss:0.648814  Test Acc:77.06%  Learning Rate:0.100000
> Train Epoch 4/20: 100%|██████████| 390/390 [00:13<00:00, 28.63it/s, Train Acc=0.817, Train Loss=0.532]
> Test Epoch 4/20: 100%|██████████| 78/78 [00:01<00:00, 61.96it/s, Test Acc=0.762, Test Loss=0.697]
> Epoch [  4/ 20]  Train Loss:0.532490  Train Acc:81.67% Test Loss:0.696845  Test Acc:76.20%  Learning Rate:0.100000
> Train Epoch 5/20: 100%|██████████| 390/390 [00:13<00:00, 28.49it/s, Train Acc=0.841, Train Loss=0.459]
> Test Epoch 5/20: 100%|██████████| 78/78 [00:01<00:00, 63.10it/s, Test Acc=0.785, Test Loss=0.642]
> Epoch [  5/ 20]  Train Loss:0.459192  Train Acc:84.09% Test Loss:0.641512  Test Acc:78.46%  Learning Rate:0.100000
> Train Epoch 6/20: 100%|██████████| 390/390 [00:13<00:00, 28.97it/s, Train Acc=0.861, Train Loss=0.406]
> Test Epoch 6/20: 100%|██████████| 78/78 [00:01<00:00, 61.06it/s, Test Acc=0.792, Test Loss=0.606]
> Epoch [  6/ 20]  Train Loss:0.405891  Train Acc:86.11% Test Loss:0.605668  Test Acc:79.21%  Learning Rate:0.100000
> Train Epoch 7/20: 100%|██████████| 390/390 [00:14<00:00, 27.13it/s, Train Acc=0.875, Train Loss=0.361]
> Test Epoch 7/20: 100%|██████████| 78/78 [00:01<00:00, 60.71it/s, Test Acc=0.756, Test Loss=0.733]
> Epoch [  7/ 20]  Train Loss:0.361291  Train Acc:87.53% Test Loss:0.732757  Test Acc:75.64%  Learning Rate:0.100000
> Train Epoch 8/20: 100%|██████████| 390/390 [00:14<00:00, 27.40it/s, Train Acc=0.883, Train Loss=0.334]
> Test Epoch 8/20: 100%|██████████| 78/78 [00:01<00:00, 58.71it/s, Test Acc=0.729, Test Loss=0.913]
> Epoch [  8/ 20]  Train Loss:0.334168  Train Acc:88.33% Test Loss:0.913161  Test Acc:72.88%  Learning Rate:0.100000
> Train Epoch 9/20: 100%|██████████| 390/390 [00:13<00:00, 28.00it/s, Train Acc=0.895, Train Loss=0.307]
> Test Epoch 9/20: 100%|██████████| 78/78 [00:01<00:00, 60.32it/s, Test Acc=0.711, Test Loss=0.933]
> Epoch [  9/ 20]  Train Loss:0.306639  Train Acc:89.53% Test Loss:0.933174  Test Acc:71.12%  Learning Rate:0.100000
> Train Epoch 10/20: 100%|██████████| 390/390 [00:13<00:00, 27.92it/s, Train Acc=0.898, Train Loss=0.288]
> Test Epoch 10/20: 100%|██████████| 78/78 [00:01<00:00, 59.58it/s, Test Acc=0.8, Test Loss=0.624]  
> Epoch [ 10/ 20]  Train Loss:0.288383  Train Acc:89.77% Test Loss:0.623786  Test Acc:80.00%  Learning Rate:0.100000
> Train Epoch 11/20: 100%|██████████| 390/390 [00:13<00:00, 28.28it/s, Train Acc=0.907, Train Loss=0.269]
> Test Epoch 11/20: 100%|██████████| 78/78 [00:01<00:00, 61.92it/s, Test Acc=0.797, Test Loss=0.651]
> Epoch [ 11/ 20]  Train Loss:0.269391  Train Acc:90.69% Test Loss:0.651273  Test Acc:79.69%  Learning Rate:0.100000
> Train Epoch 12/20: 100%|██████████| 390/390 [00:13<00:00, 27.94it/s, Train Acc=0.909, Train Loss=0.263]
> Test Epoch 12/20: 100%|██████████| 78/78 [00:01<00:00, 62.66it/s, Test Acc=0.758, Test Loss=0.799]
> Epoch [ 12/ 20]  Train Loss:0.262791  Train Acc:90.85% Test Loss:0.798698  Test Acc:75.84%  Learning Rate:0.100000
> Train Epoch 13/20: 100%|██████████| 390/390 [00:13<00:00, 28.12it/s, Train Acc=0.913, Train Loss=0.249]
> Test Epoch 13/20: 100%|██████████| 78/78 [00:01<00:00, 62.38it/s, Test Acc=0.814, Test Loss=0.572]
> Epoch [ 13/ 20]  Train Loss:0.248744  Train Acc:91.29% Test Loss:0.572026  Test Acc:81.42%  Learning Rate:0.100000
> Train Epoch 14/20: 100%|██████████| 390/390 [00:14<00:00, 26.69it/s, Train Acc=0.919, Train Loss=0.236]
> Test Epoch 14/20: 100%|██████████| 78/78 [00:01<00:00, 60.12it/s, Test Acc=0.79, Test Loss=0.683] 
> Epoch [ 14/ 20]  Train Loss:0.235874  Train Acc:91.90% Test Loss:0.683216  Test Acc:79.03%  Learning Rate:0.100000
> Train Epoch 15/20: 100%|██████████| 390/390 [00:14<00:00, 27.47it/s, Train Acc=0.923, Train Loss=0.224]
> Test Epoch 15/20: 100%|██████████| 78/78 [00:01<00:00, 61.41it/s, Test Acc=0.783, Test Loss=0.776]
> Epoch [ 15/ 20]  Train Loss:0.223898  Train Acc:92.26% Test Loss:0.776317  Test Acc:78.34%  Learning Rate:0.100000
> Train Epoch 16/20: 100%|██████████| 390/390 [00:14<00:00, 27.72it/s, Train Acc=0.922, Train Loss=0.226]
> Test Epoch 16/20: 100%|██████████| 78/78 [00:01<00:00, 62.00it/s, Test Acc=0.758, Test Loss=0.818]
> Epoch [ 16/ 20]  Train Loss:0.225781  Train Acc:92.20% Test Loss:0.818387  Test Acc:75.76%  Learning Rate:0.100000
> Train Epoch 17/20: 100%|██████████| 390/390 [00:13<00:00, 28.18it/s, Train Acc=0.927, Train Loss=0.212]
> Test Epoch 17/20: 100%|██████████| 78/78 [00:01<00:00, 60.83it/s, Test Acc=0.764, Test Loss=0.872]
> Epoch [ 17/ 20]  Train Loss:0.211836  Train Acc:92.71% Test Loss:0.872311  Test Acc:76.35%  Learning Rate:0.100000
> Train Epoch 18/20: 100%|██████████| 390/390 [00:13<00:00, 28.00it/s, Train Acc=0.929, Train Loss=0.209]
> Test Epoch 18/20: 100%|██████████| 78/78 [00:01<00:00, 62.29it/s, Test Acc=0.761, Test Loss=0.818]
> Epoch [ 18/ 20]  Train Loss:0.209121  Train Acc:92.87% Test Loss:0.818234  Test Acc:76.09%  Learning Rate:0.100000
> Train Epoch 19/20: 100%|██████████| 390/390 [00:13<00:00, 28.04it/s, Train Acc=0.928, Train Loss=0.21] 
> Test Epoch 19/20: 100%|██████████| 78/78 [00:01<00:00, 59.10it/s, Test Acc=0.764, Test Loss=0.8]  
> Epoch [ 19/ 20]  Train Loss:0.210039  Train Acc:92.76% Test Loss:0.799828  Test Acc:76.42%  Learning Rate:0.100000
> Train Epoch 20/20: 100%|██████████| 390/390 [00:14<00:00, 26.80it/s, Train Acc=0.929, Train Loss=0.207]
> Test Epoch 20/20: 100%|██████████| 78/78 [00:01<00:00, 60.36it/s, Test Acc=0.772, Test Loss=0.775]
> Epoch [ 20/ 20]  Train Loss:0.206847  Train Acc:92.89% Test Loss:0.775048  Test Acc:77.22%  Learning Rate:0.100000
> ```

### 训练曲线可视化

接着可以分别打印，损失函数曲线，准确率曲线和学习率曲线

```python
plot_history(epoch ,Acc, Loss, Lr)
```

#### 损失函数曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/8790c66354154e9f826988c2422a6928.png)



#### 准确率曲线



![在这里插入图片描述](https://img-blog.csdnimg.cn/0175d76839754031ae840937731bf2c4.png)

#### 学习率曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/a3bb60aafda84cc9ab136454ddb913ce.png)



可以运行以下代码进行可视化

```bash
tensorboard --logdir logs
```

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
> Accuracy of the network on the 10000 test images: 77.23 %
> ```

可以看到SENet的模型在测试集中准确率达到77.23%左右

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
> Accuracy of airplane : 90.07 %
> Accuracy of automobile : 90.88 %
> Accuracy of  bird : 53.56 %
> Accuracy of   cat : 54.05 %
> Accuracy of  deer : 70.97 %
> Accuracy of   dog : 72.17 %
> Accuracy of  frog : 88.09 %
> Accuracy of horse : 70.44 %
> Accuracy of  ship : 93.89 %
> Accuracy of truck : 88.00 %
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
> Accuracy Rate = 81.25%
> ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/f7517b2746e842e78b45bacd1793edb0.png)

## 8. 保存模型

```python
torch.save(net,save_path[:-4]+'_'+str(epoch)+'.pth')
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
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetV2(num_classes=10)

model = torch.load(save_path, map_location="cpu")  # 加载模型
model.to(device)
model.eval()  # 把模型转为test模式
```

并且为了方便，定义了一个predict函数，简单思想就是，先resize成网络使用的shape，然后进行变化tensor输入即可，不过这里有一个点，我们需要对我们的图片也进行transforms，因为我们的训练的时候，对每个图像也是进行了transforms的，所以我们需要保持一致



```python
def predict(img):
    trans = transforms.Compose([transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                 std=(0.5, 0.5, 0.5)),
                           ])
 
    img = trans(img)
    img = img.to(device)
    # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    img = img.unsqueeze(0)  # 扩展后，为[1，3，32，32]
    output = model(img)
    prob = F.softmax(output,dim=1) #prob是10个分类的概率
    print("概率",prob)
    value, predicted = torch.max(output.data, 1)
    print("类别",predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print("分类",pred_class)
```

```bash
# 读取要预测的图片
img = Image.open("./airplane.jpg").convert('RGB') # 读取图像
img
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/a4baf7e595b54cc1b3221e57b62ec43d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)



```python
predict(img)
```

```bash
概率 tensor([[9.9239e-01, 2.1051e-05, 7.0973e-05, 2.2179e-06, 2.0383e-06, 1.1027e-07,
         1.9355e-06, 5.0448e-08, 7.4134e-03, 1.0011e-04]], device='cuda:0',
       grad_fn=<SoftmaxBackward0>)
类别 0
tensor([10.6461], device='cuda:0')
分类 plane
```

这里就可以看到，我们最后的结果，分类为plane，我们的置信率大概是99.2%，看起来是很不错的

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
predict(img)
```

> ```python
> 概率 tensor([[0.5546, 0.0017, 0.1146, 0.1129, 0.0331, 0.1359, 0.0067, 0.0070, 0.0271,
>             0.0064]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
> 类别 0
> tensor([2.9417], device='cuda:0')
> 分类 plane
> ```

可以看到，分类不正确了，对于cat还是分类到了airplane，这有可能是我们只迭代了20次的原因，对真实的cat的图片没有那么敏感，可以增加迭代次数，调参肯定能得到更好的结果。



## 10.总结

最后对SENet进行一个总结，SENet（Squeeze-and-Excitation Network主要优点如下：

1. 提升模型性能：SENet能够充分利用不同通道的特征信息，通过自适应地调整通道的权重，从而提高模型的性能。实验结果表明，SENet在多个图像分类任务上都能取得比其他模型更好的性能。
2. 灵活性高：SENet的结构可以很容易地嵌入到现有的卷积神经网络中，而不需要重新设计整个网络。这使得SENet可以方便地应用于各种计算机视觉任务，从而提高模型的效果。
3. 参数量较小：SENet中引入的特征重标定模块（Squeeze-and-Excitation）只需要很少的参数量，而且可以与现有的网络结构相结合，从而不会增加太多的计算负担和模型复杂度。
4. 可解释性强：SENet的特征重标定模块具有很强的可解释性。通过对每个通道的权重进行调整，SENet能够更好地理解网络中不同通道的特征对于任务的贡献，从而提高了模型的可解释性。这也使得SENet在一些需要解释模型决策的场景下具有优势。

综上所述，SENet在提高模型性能、灵活性、参数量和可解释性方面都具有优点，使得它成为一种受欢迎的卷积神经网络结构。

顺带提一句，我们的数据和代码都在我的汇总篇里有说明，如果需要，可以自取

这里再贴一下汇总篇：[汇总篇](https://redamancy.blog.csdn.net/article/details/127859300)



