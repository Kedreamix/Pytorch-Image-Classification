# Pytorch CIFAR10图像分类 GoogLeNet篇

[toc]

这里贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)

## 4.定义网络（GoogLeNet）

GoogLeNet在2014年由Google团队提出（与VGG网络同年，注意GoogLeNet中的L大写是为了致敬LeNet），斩获当年ImageNet竞赛中Classification Task (分类任务) 第一名。原论文名称是《Going deeper with convolutions》，下面是网络结构图。

![](img/Inception.jpg)

**说说该网络中的亮点：**

（1）引入了Inception结构（融合不同尺度的特征信息）

（2）使用1x1的卷积核进行降维以及映射处理 （虽然VGG网络中也有，但该论文介绍的更详细）

（3）添加两个辅助分类器帮助训练

（4）丢弃全连接层，使用平均池化层（大大减少模型参数，除去两个辅助分类器，网络大小只有vgg的1/20）

接着我们来分析一下Inception结构：

Inception v1网络是一个精心设计的22层卷积网络，并提出了具有良好局部特征结构Inception模块，即对特征并行地执行多个大小不同的卷积运算与池化，最后再拼接到一起。由于1×1、3×3和5×5的卷积运算对应不同的特征图区域，因此这样做的好处是可以得到更好的图像表征信息。为了让四个分支的输出能够在深度方向进行拼接，必须保证四个分支输出的特征矩阵高度和宽度都相同）。

Inception模块如图所示，使用了三个不同大小的卷积核进行卷积运算，同时还有一个最大值池化，然后将这4部分级联起来（通道拼接），送入下一层。

![](img\Inception2.png)

分支1是卷积核大小为1x1的卷积层，stride=1，

分支2是卷积核大小为3x3的卷积层，stride=1，padding=1（保证输出特征矩阵的高和宽和输入特征矩阵相等），

分支3是卷积核大小为5x5的卷积层，stride=1，padding=2（保证输出特征矩阵的高和宽和输入特征矩阵相等），

分支4是池化核大小为3x3的最大池化下采样，stride=1，padding=1（保证输出特征矩阵的高和宽和输入特征矩阵相等）。

在上述模块的基础上，为进一步降低网络参数量，Inception又增加了多个1×1的卷积模块。如图3.14所示，这种1×1的模块可以先将特征图降维，再送给3×3和5×5大小的卷积核，由于通道数的降低，参数量也有了较大的减少。

Inception v1网络一共有9个上述堆叠的模块，共有22层，在最后的Inception模块处使用了全局平均池化

![](img\Inception3.png)

为了避免深层网络训练时带来的梯度消失问题，作者还引入了两个辅助的分类器，在第3个与第6个Inception模块输出后执行Softmax并计算损失，在训练时和最后的损失一并回传。
接着下来在看看辅助分类器结构，网络中的两个辅助分类器结构是一模一样的，如下图所示：

![](img\GooLeNet2.png)

辅助分类器：

第一层是一个平均池化下采样层，池化核大小为5x5，stride=3

第二层是卷积层，卷积核大小为1x1，stride=1，卷积核个数是128

第三层是全连接层，节点个数是1024

第四层是全连接层，节点个数是1000（对应分类的类别个数）

Inception v1的参数量是AlexNet的1/12，VGGNet的1/3，适合处理大规模数据，尤其是对于计算资源有限的平台。

**Inception v2**

在Inception v1网络的基础上，随后又出现了多个Inception版本。Inception v2进一步通过卷积分解与正则化实现更高效的计算，增加了BN层，同时利用两个级联的3×3卷积取代Inception v1版本中的5×5卷积，如图3.15所示，这种方式既减少了卷积参数量，也增加了网络的非线性能力。

![](img\Inceptionv2.png)

此外除了这两个版本，这几年还分别出了Inception v3和Inception v4。

Inception v3在Inception v2的基础上，使用了RMSProp优化器，在辅助的分类器部分增加了7×7的卷积，并且使用了标签平滑技术。


Inception v4则是将Inception的思想与残差网络进行了结合，显著提升了训练速度与模型准确率，这里对于模块细节不再展开讲述。至于残差网络这一里程碑式的结构，正是由下一节的网络ResNet引出的。

**这里演示只展示一个Inception v1的网络结构**有兴趣的话也可以尝试其他网络结构



首先我们还是得判断是否可以利用GPU，因为GPU的速度可能会比我们用CPU的速度快20-50倍左右，特别是对卷积神经网络来说，更是提升特别明显。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
# 定义一个卷积加一个 relu 激活函数和一个 batchnorm 作为一个基本的层结构
class BasicConv2d(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.batch = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
```

```python
# 首先定义Inception
class Inception(nn.Module):
    def __init__(self, in_channel, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj):
        super(Inception, self).__init__()
        
        # 第一条线路
        self.branch1x1 = BasicConv2d(in_channel, n1x1, 1)
        
        # 第二条线路
        self.branch3x3 = nn.Sequential( 
            BasicConv2d(in_channel, n3x3red, 1),
            BasicConv2d(n3x3red, n3x3, 3, padding=1)
        )
        
        # 第三条线路
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channel, n5x5red, 1),
            BasicConv2d(n5x5red, n5x5, 5, padding=2)
        )
        
        # 第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size= 3, stride=1, padding=1),
            BasicConv2d(in_channel, pool_proj, 1)
        )
    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output
```

```python
class InceptionAux(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(InceptionAux, self).__init__()
#         self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.averagePool = nn.AvgPool2d(kernel_size=2)
        self.conv = BasicConv2d(in_channel, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        
        self.fc1 = nn.Sequential(
#             nn.Linear(2048, 1024),
            nn.Linear(128,64),
            nn.ReLU(True)
        )
#         self.drop = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, num_classes)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):

        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc1(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x
```

我们可以测试一下Inception，是否会改变

```python
test_net = Inception(3, 64, 48, 64, 64, 96, 32)
test_x =torch.zeros(1, 3, 32, 32)
print('input shape: {} x {} x {}'.format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))
```

> ```python
> input shape: 3 x 32 x 32
> output shape: 256 x 32 x 32
> ```

```python
class GoogLeNet(nn.Module):
    def __init__(self, in_channel, num_classes, aux_logits=False, verbose=False, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.verbose = verbose
        self.aux_logits = aux_logits
        
#         self.block1 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel=64, kernel=7, stride=2, padding=3),
#             nn.MaxPool2d(3, 2, ceil_mode=True)
#         )
        
#         self.block2 = nn.Sequential(
#             BasicConv2d(64, 64, kernel=1),
#             BasicConv2d(64, 192, kernel=3, padding=1),
#             nn.MaxPool2d(3, 2, ceil_mode=True)
#         )
        
#         self.block3 = nn.Sequential(
#             inception(192, 64, 96, 128, 16, 32, 32),
#             inception(256, 128, 128, 192, 32, 96, 64),
#             nn.MaxPool2d(3, 2, ceil_mode=True)
#         )
        
#         self.block4 = nn.Sequential(
#             inception(480, 192, 96, 208, 16, 48, 64),
#             inception(512, 160, 112, 224, 24, 64, 64),
#             inception(512, 128, 128, 256, 24, 64, 64),
#             inception(512, 112, 144, 288, 32, 64, 64),
#             inception(528, 256, 160, 320, 32, 128, 128),
#             nn.MaxPool2d(3, 2, ceil_mode=True)
#         )
        
#         self.block5 = nn.Sequential(
#             inception(832, 256, 160, 320, 32, 128, 128),
#             inception(832, 384, 182, 384, 48, 128, 128),
            
#         )
        # block1
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # block2
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # block3
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # block4
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # block5
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        
#         self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AvgPool2d(1) # 对32x32 不一样
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()
        
            
    def forward(self, x):
#         x = self.block1(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
#         x = self.block2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
#         x = self.block3(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
#         x = self.block4(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)
            if self.verbose:
                print('aux 1 output: {}'.format(aux1.shape))
            
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)
            if self.verbose:
                print('aux 2 output: {}'.format(aux2.shape))
        x = self.inception4e(x)
        x = self.maxpool4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
#         x = self.block5(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.classifier(x)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
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
net = GoogLeNet(3,10,aux_logits = True, verbose = False).to(device)
```

```python
summary(net,(3,32,32))
```

> ```python
> ----------------------------------------------------------------
>         Layer (type)               Output Shape         Param #
> ================================================================
>             Conv2d-1           [-1, 64, 16, 16]           9,472
>        BatchNorm2d-2           [-1, 64, 16, 16]             128
>               ReLU-3           [-1, 64, 16, 16]               0
>        BasicConv2d-4           [-1, 64, 16, 16]               0
>          MaxPool2d-5             [-1, 64, 8, 8]               0
>             Conv2d-6             [-1, 64, 8, 8]           4,160
>        BatchNorm2d-7             [-1, 64, 8, 8]             128
>               ReLU-8             [-1, 64, 8, 8]               0
>        BasicConv2d-9             [-1, 64, 8, 8]               0
>            Conv2d-10            [-1, 192, 8, 8]         110,784
>       BatchNorm2d-11            [-1, 192, 8, 8]             384
>              ReLU-12            [-1, 192, 8, 8]               0
>       BasicConv2d-13            [-1, 192, 8, 8]               0
>         MaxPool2d-14            [-1, 192, 4, 4]               0
>            Conv2d-15             [-1, 64, 4, 4]          12,352
>       BatchNorm2d-16             [-1, 64, 4, 4]             128
>              ReLU-17             [-1, 64, 4, 4]               0
>       BasicConv2d-18             [-1, 64, 4, 4]               0
>            Conv2d-19             [-1, 96, 4, 4]          18,528
>       BatchNorm2d-20             [-1, 96, 4, 4]             192
>              ReLU-21             [-1, 96, 4, 4]               0
>       BasicConv2d-22             [-1, 96, 4, 4]               0
>            Conv2d-23            [-1, 128, 4, 4]         110,720
>       BatchNorm2d-24            [-1, 128, 4, 4]             256
>              ReLU-25            [-1, 128, 4, 4]               0
>       BasicConv2d-26            [-1, 128, 4, 4]               0
>            Conv2d-27             [-1, 16, 4, 4]           3,088
>       BatchNorm2d-28             [-1, 16, 4, 4]              32
>              ReLU-29             [-1, 16, 4, 4]               0
>       BasicConv2d-30             [-1, 16, 4, 4]               0
>            Conv2d-31             [-1, 32, 4, 4]          12,832
>       BatchNorm2d-32             [-1, 32, 4, 4]              64
>              ReLU-33             [-1, 32, 4, 4]               0
>       BasicConv2d-34             [-1, 32, 4, 4]               0
>         MaxPool2d-35            [-1, 192, 4, 4]               0
>            Conv2d-36             [-1, 32, 4, 4]           6,176
>       BatchNorm2d-37             [-1, 32, 4, 4]              64
>              ReLU-38             [-1, 32, 4, 4]               0
>       BasicConv2d-39             [-1, 32, 4, 4]               0
>         Inception-40            [-1, 256, 4, 4]               0
>            Conv2d-41            [-1, 128, 4, 4]          32,896
>       BatchNorm2d-42            [-1, 128, 4, 4]             256
>              ReLU-43            [-1, 128, 4, 4]               0
>       BasicConv2d-44            [-1, 128, 4, 4]               0
>            Conv2d-45            [-1, 128, 4, 4]          32,896
>       BatchNorm2d-46            [-1, 128, 4, 4]             256
>              ReLU-47            [-1, 128, 4, 4]               0
>       BasicConv2d-48            [-1, 128, 4, 4]               0
>            Conv2d-49            [-1, 192, 4, 4]         221,376
>       BatchNorm2d-50            [-1, 192, 4, 4]             384
>              ReLU-51            [-1, 192, 4, 4]               0
>       BasicConv2d-52            [-1, 192, 4, 4]               0
>            Conv2d-53             [-1, 32, 4, 4]           8,224
>       BatchNorm2d-54             [-1, 32, 4, 4]              64
>              ReLU-55             [-1, 32, 4, 4]               0
>       BasicConv2d-56             [-1, 32, 4, 4]               0
>            Conv2d-57             [-1, 96, 4, 4]          76,896
>       BatchNorm2d-58             [-1, 96, 4, 4]             192
>              ReLU-59             [-1, 96, 4, 4]               0
>       BasicConv2d-60             [-1, 96, 4, 4]               0
>         MaxPool2d-61            [-1, 256, 4, 4]               0
>            Conv2d-62             [-1, 64, 4, 4]          16,448
>       BatchNorm2d-63             [-1, 64, 4, 4]             128
>              ReLU-64             [-1, 64, 4, 4]               0
>       BasicConv2d-65             [-1, 64, 4, 4]               0
>         Inception-66            [-1, 480, 4, 4]               0
>         MaxPool2d-67            [-1, 480, 2, 2]               0
>            Conv2d-68            [-1, 192, 2, 2]          92,352
>       BatchNorm2d-69            [-1, 192, 2, 2]             384
>              ReLU-70            [-1, 192, 2, 2]               0
>       BasicConv2d-71            [-1, 192, 2, 2]               0
>            Conv2d-72             [-1, 96, 2, 2]          46,176
>       BatchNorm2d-73             [-1, 96, 2, 2]             192
>              ReLU-74             [-1, 96, 2, 2]               0
>       BasicConv2d-75             [-1, 96, 2, 2]               0
>            Conv2d-76            [-1, 208, 2, 2]         179,920
>       BatchNorm2d-77            [-1, 208, 2, 2]             416
>              ReLU-78            [-1, 208, 2, 2]               0
>       BasicConv2d-79            [-1, 208, 2, 2]               0
>            Conv2d-80             [-1, 16, 2, 2]           7,696
>       BatchNorm2d-81             [-1, 16, 2, 2]              32
>              ReLU-82             [-1, 16, 2, 2]               0
>       BasicConv2d-83             [-1, 16, 2, 2]               0
>            Conv2d-84             [-1, 48, 2, 2]          19,248
>       BatchNorm2d-85             [-1, 48, 2, 2]              96
>              ReLU-86             [-1, 48, 2, 2]               0
>       BasicConv2d-87             [-1, 48, 2, 2]               0
>         MaxPool2d-88            [-1, 480, 2, 2]               0
>            Conv2d-89             [-1, 64, 2, 2]          30,784
>       BatchNorm2d-90             [-1, 64, 2, 2]             128
>              ReLU-91             [-1, 64, 2, 2]               0
>       BasicConv2d-92             [-1, 64, 2, 2]               0
>         Inception-93            [-1, 512, 2, 2]               0
>         AvgPool2d-94            [-1, 512, 1, 1]               0
>            Conv2d-95            [-1, 128, 1, 1]          65,664
>       BatchNorm2d-96            [-1, 128, 1, 1]             256
>              ReLU-97            [-1, 128, 1, 1]               0
>       BasicConv2d-98            [-1, 128, 1, 1]               0
>            Linear-99                   [-1, 64]           8,256
>             ReLU-100                   [-1, 64]               0
>           Linear-101                   [-1, 10]             650
>     InceptionAux-102                   [-1, 10]               0
>           Conv2d-103            [-1, 160, 2, 2]          82,080
>      BatchNorm2d-104            [-1, 160, 2, 2]             320
>             ReLU-105            [-1, 160, 2, 2]               0
>      BasicConv2d-106            [-1, 160, 2, 2]               0
>           Conv2d-107            [-1, 112, 2, 2]          57,456
>      BatchNorm2d-108            [-1, 112, 2, 2]             224
>             ReLU-109            [-1, 112, 2, 2]               0
>      BasicConv2d-110            [-1, 112, 2, 2]               0
>           Conv2d-111            [-1, 224, 2, 2]         226,016
>      BatchNorm2d-112            [-1, 224, 2, 2]             448
>             ReLU-113            [-1, 224, 2, 2]               0
>      BasicConv2d-114            [-1, 224, 2, 2]               0
>           Conv2d-115             [-1, 24, 2, 2]          12,312
>      BatchNorm2d-116             [-1, 24, 2, 2]              48
>             ReLU-117             [-1, 24, 2, 2]               0
>      BasicConv2d-118             [-1, 24, 2, 2]               0
>           Conv2d-119             [-1, 64, 2, 2]          38,464
>      BatchNorm2d-120             [-1, 64, 2, 2]             128
>             ReLU-121             [-1, 64, 2, 2]               0
>      BasicConv2d-122             [-1, 64, 2, 2]               0
>        MaxPool2d-123            [-1, 512, 2, 2]               0
>           Conv2d-124             [-1, 64, 2, 2]          32,832
>      BatchNorm2d-125             [-1, 64, 2, 2]             128
>             ReLU-126             [-1, 64, 2, 2]               0
>      BasicConv2d-127             [-1, 64, 2, 2]               0
>        Inception-128            [-1, 512, 2, 2]               0
>           Conv2d-129            [-1, 128, 2, 2]          65,664
>      BatchNorm2d-130            [-1, 128, 2, 2]             256
>             ReLU-131            [-1, 128, 2, 2]               0
>      BasicConv2d-132            [-1, 128, 2, 2]               0
>           Conv2d-133            [-1, 128, 2, 2]          65,664
>      BatchNorm2d-134            [-1, 128, 2, 2]             256
>             ReLU-135            [-1, 128, 2, 2]               0
>      BasicConv2d-136            [-1, 128, 2, 2]               0
>           Conv2d-137            [-1, 256, 2, 2]         295,168
>      BatchNorm2d-138            [-1, 256, 2, 2]             512
>             ReLU-139            [-1, 256, 2, 2]               0
>      BasicConv2d-140            [-1, 256, 2, 2]               0
>           Conv2d-141             [-1, 24, 2, 2]          12,312
>      BatchNorm2d-142             [-1, 24, 2, 2]              48
>             ReLU-143             [-1, 24, 2, 2]               0
>      BasicConv2d-144             [-1, 24, 2, 2]               0
>           Conv2d-145             [-1, 64, 2, 2]          38,464
>      BatchNorm2d-146             [-1, 64, 2, 2]             128
>             ReLU-147             [-1, 64, 2, 2]               0
>      BasicConv2d-148             [-1, 64, 2, 2]               0
>        MaxPool2d-149            [-1, 512, 2, 2]               0
>           Conv2d-150             [-1, 64, 2, 2]          32,832
>      BatchNorm2d-151             [-1, 64, 2, 2]             128
>             ReLU-152             [-1, 64, 2, 2]               0
>      BasicConv2d-153             [-1, 64, 2, 2]               0
>        Inception-154            [-1, 512, 2, 2]               0
>           Conv2d-155            [-1, 112, 2, 2]          57,456
>      BatchNorm2d-156            [-1, 112, 2, 2]             224
>             ReLU-157            [-1, 112, 2, 2]               0
>      BasicConv2d-158            [-1, 112, 2, 2]               0
>           Conv2d-159            [-1, 144, 2, 2]          73,872
>      BatchNorm2d-160            [-1, 144, 2, 2]             288
>             ReLU-161            [-1, 144, 2, 2]               0
>      BasicConv2d-162            [-1, 144, 2, 2]               0
>           Conv2d-163            [-1, 288, 2, 2]         373,536
>      BatchNorm2d-164            [-1, 288, 2, 2]             576
>             ReLU-165            [-1, 288, 2, 2]               0
>      BasicConv2d-166            [-1, 288, 2, 2]               0
>           Conv2d-167             [-1, 32, 2, 2]          16,416
>      BatchNorm2d-168             [-1, 32, 2, 2]              64
>             ReLU-169             [-1, 32, 2, 2]               0
>      BasicConv2d-170             [-1, 32, 2, 2]               0
>           Conv2d-171             [-1, 64, 2, 2]          51,264
>      BatchNorm2d-172             [-1, 64, 2, 2]             128
>             ReLU-173             [-1, 64, 2, 2]               0
>      BasicConv2d-174             [-1, 64, 2, 2]               0
>        MaxPool2d-175            [-1, 512, 2, 2]               0
>           Conv2d-176             [-1, 64, 2, 2]          32,832
>      BatchNorm2d-177             [-1, 64, 2, 2]             128
>             ReLU-178             [-1, 64, 2, 2]               0
>      BasicConv2d-179             [-1, 64, 2, 2]               0
>        Inception-180            [-1, 528, 2, 2]               0
>        AvgPool2d-181            [-1, 528, 1, 1]               0
>           Conv2d-182            [-1, 128, 1, 1]          67,712
>      BatchNorm2d-183            [-1, 128, 1, 1]             256
>             ReLU-184            [-1, 128, 1, 1]               0
>      BasicConv2d-185            [-1, 128, 1, 1]               0
>           Linear-186                   [-1, 64]           8,256
>             ReLU-187                   [-1, 64]               0
>           Linear-188                   [-1, 10]             650
>     InceptionAux-189                   [-1, 10]               0
>           Conv2d-190            [-1, 256, 2, 2]         135,424
>      BatchNorm2d-191            [-1, 256, 2, 2]             512
>             ReLU-192            [-1, 256, 2, 2]               0
>      BasicConv2d-193            [-1, 256, 2, 2]               0
>           Conv2d-194            [-1, 160, 2, 2]          84,640
>      BatchNorm2d-195            [-1, 160, 2, 2]             320
>             ReLU-196            [-1, 160, 2, 2]               0
>      BasicConv2d-197            [-1, 160, 2, 2]               0
>           Conv2d-198            [-1, 320, 2, 2]         461,120
>      BatchNorm2d-199            [-1, 320, 2, 2]             640
>             ReLU-200            [-1, 320, 2, 2]               0
>      BasicConv2d-201            [-1, 320, 2, 2]               0
>           Conv2d-202             [-1, 32, 2, 2]          16,928
>      BatchNorm2d-203             [-1, 32, 2, 2]              64
>             ReLU-204             [-1, 32, 2, 2]               0
>      BasicConv2d-205             [-1, 32, 2, 2]               0
>           Conv2d-206            [-1, 128, 2, 2]         102,528
>      BatchNorm2d-207            [-1, 128, 2, 2]             256
>             ReLU-208            [-1, 128, 2, 2]               0
>      BasicConv2d-209            [-1, 128, 2, 2]               0
>        MaxPool2d-210            [-1, 528, 2, 2]               0
>           Conv2d-211            [-1, 128, 2, 2]          67,712
>      BatchNorm2d-212            [-1, 128, 2, 2]             256
>             ReLU-213            [-1, 128, 2, 2]               0
>      BasicConv2d-214            [-1, 128, 2, 2]               0
>        Inception-215            [-1, 832, 2, 2]               0
>        MaxPool2d-216            [-1, 832, 1, 1]               0
>           Conv2d-217            [-1, 256, 1, 1]         213,248
>      BatchNorm2d-218            [-1, 256, 1, 1]             512
>             ReLU-219            [-1, 256, 1, 1]               0
>      BasicConv2d-220            [-1, 256, 1, 1]               0
>           Conv2d-221            [-1, 160, 1, 1]         133,280
>      BatchNorm2d-222            [-1, 160, 1, 1]             320
>             ReLU-223            [-1, 160, 1, 1]               0
>      BasicConv2d-224            [-1, 160, 1, 1]               0
>           Conv2d-225            [-1, 320, 1, 1]         461,120
>      BatchNorm2d-226            [-1, 320, 1, 1]             640
>             ReLU-227            [-1, 320, 1, 1]               0
>      BasicConv2d-228            [-1, 320, 1, 1]               0
>           Conv2d-229             [-1, 32, 1, 1]          26,656
>      BatchNorm2d-230             [-1, 32, 1, 1]              64
>             ReLU-231             [-1, 32, 1, 1]               0
>      BasicConv2d-232             [-1, 32, 1, 1]               0
>           Conv2d-233            [-1, 128, 1, 1]         102,528
>      BatchNorm2d-234            [-1, 128, 1, 1]             256
>             ReLU-235            [-1, 128, 1, 1]               0
>      BasicConv2d-236            [-1, 128, 1, 1]               0
>        MaxPool2d-237            [-1, 832, 1, 1]               0
>           Conv2d-238            [-1, 128, 1, 1]         106,624
>      BatchNorm2d-239            [-1, 128, 1, 1]             256
>             ReLU-240            [-1, 128, 1, 1]               0
>      BasicConv2d-241            [-1, 128, 1, 1]               0
>        Inception-242            [-1, 832, 1, 1]               0
>           Conv2d-243            [-1, 384, 1, 1]         319,872
>      BatchNorm2d-244            [-1, 384, 1, 1]             768
>             ReLU-245            [-1, 384, 1, 1]               0
>      BasicConv2d-246            [-1, 384, 1, 1]               0
>           Conv2d-247            [-1, 192, 1, 1]         159,936
>      BatchNorm2d-248            [-1, 192, 1, 1]             384
>             ReLU-249            [-1, 192, 1, 1]               0
>      BasicConv2d-250            [-1, 192, 1, 1]               0
>           Conv2d-251            [-1, 384, 1, 1]         663,936
>      BatchNorm2d-252            [-1, 384, 1, 1]             768
>             ReLU-253            [-1, 384, 1, 1]               0
>      BasicConv2d-254            [-1, 384, 1, 1]               0
>           Conv2d-255             [-1, 48, 1, 1]          39,984
>      BatchNorm2d-256             [-1, 48, 1, 1]              96
>             ReLU-257             [-1, 48, 1, 1]               0
>      BasicConv2d-258             [-1, 48, 1, 1]               0
>           Conv2d-259            [-1, 128, 1, 1]         153,728
>      BatchNorm2d-260            [-1, 128, 1, 1]             256
>             ReLU-261            [-1, 128, 1, 1]               0
>      BasicConv2d-262            [-1, 128, 1, 1]               0
>        MaxPool2d-263            [-1, 832, 1, 1]               0
>           Conv2d-264            [-1, 128, 1, 1]         106,624
>      BatchNorm2d-265            [-1, 128, 1, 1]             256
>             ReLU-266            [-1, 128, 1, 1]               0
>      BasicConv2d-267            [-1, 128, 1, 1]               0
>        Inception-268           [-1, 1024, 1, 1]               0
>        AvgPool2d-269           [-1, 1024, 1, 1]               0
>          Dropout-270                 [-1, 1024]               0
>           Linear-271                   [-1, 10]          10,250
> ================================================================
> Total params: 6,150,062
> Trainable params: 6,150,062
> Non-trainable params: 0
> ----------------------------------------------------------------
> Input size (MB): 0.01
> Forward/backward pass size (MB): 2.46
> Params size (MB): 23.46
> Estimated Total Size (MB): 25.93
> ----------------------------------------------------------------
> ```

首先从我们summary可以看到，我们定义的模型的参数大概是6 millions，我们输入的是（batch，3，32，32）的张量，并且这里也能看到每一层后我们的图像输出大小的变化，最后输出10个参数，再通过softmax函数就可以得到我们每个类别的概率了。

我们也可以打印出我们的模型观察一下

```python
GoogLeNet(
  (conv1): BasicConv2d(
    (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (conv2): BasicConv2d(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (conv3): BasicConv2d(
    (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (batch): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception3a): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (inception3b): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (maxpool3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception4a): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (inception4b): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (inception4c): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (inception4d): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (inception4e): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (maxpool4): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
  (inception5a): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (inception5b): Inception(
    (branch1x1): BasicConv2d(
      (conv): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (branch3x3): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (batch): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch5x5): Sequential(
      (0): BasicConv2d(
        (conv): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicConv2d(
        (conv): Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): BasicConv2d(
        (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
        (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (aux1): InceptionAux(
    (averagePool): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (conv): BasicConv2d(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (fc1): Sequential(
      (0): Linear(in_features=128, out_features=64, bias=True)
      (1): ReLU(inplace=True)
    )
    (fc2): Linear(in_features=64, out_features=10, bias=True)
  )
  (aux2): InceptionAux(
    (averagePool): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (conv): BasicConv2d(
      (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
      (batch): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (fc1): Sequential(
      (0): Linear(in_features=128, out_features=64, bias=True)
      (1): ReLU(inplace=True)
    )
    (fc2): Linear(in_features=64, out_features=10, bias=True)
  )
  (avgpool): AvgPool2d(kernel_size=1, stride=1, padding=0)
  (dropout): Dropout(p=0.4, inplace=False)
  (classifier): Linear(in_features=1024, out_features=10, bias=True)
)
```

我们也可以测试一下输出

```python
test_net = GoogLeNet(3, 10, aux_logits = True,verbose = True)
test_x = torch.zeros(3, 3, 32, 32)
test_net.train
test_y = test_net(test_x)
print('output: {}'.format(test_y[0].shape))
```

> ```python
> block 1 output: torch.Size([3, 64, 8, 8])
> block 2 output: torch.Size([3, 192, 4, 4])
> block 3 output: torch.Size([3, 480, 2, 2])
> aux 1 output: torch.Size([3, 10])
> aux 2 output: torch.Size([3, 10])
> block 4 output: torch.Size([3, 832, 1, 1])
> block 5 output: torch.Size([3, 1024, 1, 1])
> output: torch.Size([3, 10])
> ```

如果你的电脑有多个GPU，这段代码可以利用GPU进行并行计算，加快运算速度

```python
net =LeNet5().to(device)
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
save_path = './model/GoogLeNet.pth'
```

我定义了一个train函数，在train函数中进行一个训练，并保存我们训练后的模型

```python
import time
best_acc = 0
train_acc_list, test_acc_list = [],[]
train_loss_list, test_loss_list = [],[]
lr_list  = []
for i in range(epoch):
    start = time.time()
    train_loss = 0
    train_acc = 0
    test_loss = 0
    test_acc = 0
    if torch.cuda.is_available():
        net = net.to(device)
    net.train()
    for step,data in enumerate(trainloader,start=0):
        im,label = data
        im = im.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        logits, aux_logits2, aux_logits1 = net(im)
        loss0 = criterion(logits, label)
        loss1 = criterion(aux_logits1, label)
        loss2 = criterion(aux_logits2, label)
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()
        outputs = logits
        train_loss += loss.data
        probs, pred_y = outputs.data.max(dim=1) # 得到概率
        train_acc += (pred_y==label).sum()/label.size(0)

        rate = (step + 1) / len(trainloader)
        a = "*" * int(rate * 50)
        b = "." * (50 - int(rate * 50))
        print('\r train {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoch,int(rate*100),a,b),end='')

    train_loss = train_loss / len(trainloader)
    train_acc = train_acc * 100 / len(trainloader)
#     print('train_loss:{:.3f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')  
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    lr = optimizer.param_groups[0]['lr']
    lr_list.append(lr)
    scheduler.step(train_loss)
    
    net.eval()
    with torch.no_grad():
        for step,data in enumerate(testloader,start=0):
            im,label = data
            im = im.to(device)
            label = label.to(device)
            outputs = net(im)
            loss = criterion(outputs,label)
            test_loss += loss.data
            probs, pred_y = outputs.data.max(dim=1) # 得到概率
            test_acc += (pred_y==label).sum()/label.size(0)
            rate = (step + 1) / len(testloader)
            a = "*" * int(rate * 50)
            b = "." * (50 - int(rate * 50))
            print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoch,int(rate*100),a,b),end='')
    test_loss = test_loss / len(testloader)
    test_acc = test_acc * 100 /len(testloader)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    end = time.time()
    print('\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epoch, train_loss, train_acc, test_loss, test_acc,lr), end='')
    time_ = int(end - start)
    time_ = int(end - start)
    h = time_ / 3600
    m = time_ % 3600 /60
    s = time_ % 60
    time_str = "\tTime %02d:%02d" % ( m, s)
    # 打印所用时间
    print(time_str)
    # 如果取得更好的准确率，就保存模型
    if test_acc > best_acc:
        torch.save(net,save_path)
        best_acc = test_acc

Acc = {}
Loss = {}
Acc['train_acc'] = train_acc_list
Acc['test_acc'] = test_acc_list
Loss['train_loss'] = train_loss_list
Loss['test_loss'] = test_loss_list
Lr = lr_list
```

接着可以分别打印，损失函数曲线，准确率曲线和学习率曲线

```python
plot_history(epoch ,Acc, Loss, Lr)
```

### 损失函数曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/b56aa6b5066747ffa49d72e845af6b90.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

### 准确率曲线



![在这里插入图片描述](https://img-blog.csdnimg.cn/dc9505801bff4fb88b9826aebebf6a3e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

### 学习率曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/f29480a06cdd451989d0cf62cc2e3bab.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)



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
> Accuracy of the network on the 10000 test images: 69.88 %
> ```

可以看到自定义网络的模型在测试集中准确率达到69.88%



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
> Accuracy of airplane : 86.70 %
> Accuracy of automobile : 93.80 %
> Accuracy of  bird : 56.10 %
> Accuracy of   cat : 49.50 %
> Accuracy of  deer : 79.30 %
> Accuracy of   dog : 61.20 %
> Accuracy of  frog : 86.50 %
> Accuracy of horse : 54.90 %
> Accuracy of  ship : 84.60 %
> Accuracy of truck : 48.30 %
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
> Accuracy Rate = 69.140625%
> <Figure size 1800x288 with 0 Axes>
> ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/3a62c3237cf54a149f8982019f8cd509.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 8. 保存模型

```python
torch.save(net,save_path[:-4]+str(epoch)+'.pth')
# torch.save(net, './model/GoogLeNet-256.pth')
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

model = GoogLeNet(3,num_classes=10, aux_logits=True)

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
> 概率 tensor([[9.9385e-01, 1.7306e-04, 7.7972e-04, 5.7594e-05, 9.7720e-05, 1.8072e-05,
>          2.5149e-05, 4.5894e-05, 4.7694e-03, 1.8658e-04]], device='cuda:0',
>        grad_fn=<SoftmaxBackward>)
> 类别 0
> tensor([8.0125], device='cuda:0')
> 分类 plane
> ```

这里就可以看到，我们最后的结果，分类为plane，我们的置信率大概是99.38%

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
> 概率 tensor([[0.7752, 0.0063, 0.0506, 0.0376, 0.0050, 0.0109, 0.0094, 0.0070, 0.0901,
>          0.0080]], device='cuda:0', grad_fn=<SoftmaxBackward>)
> 类别 0
> tensor([3.5706], device='cuda:0')
> 分类 plane
> ```

可以看到，我们分类的结果是plane，而真正的分类是cat，所以这个模型还不是特别完美，还需要不断完善



## 10.总结

其实随着时间的发展，Inception已经出现了多个版本，但是我们这里是最原始的Inceptionv1版本

- v1：最早的版本
- v2：加入 batch normalization 加快训练
- v3：对 inception 模块做了调整
- v4：基于 ResNet 加入了 残差连接


顺带提一句，我们的数据和代码都在我的汇总篇里有说明，如果需要，可以自取

这里再贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)
