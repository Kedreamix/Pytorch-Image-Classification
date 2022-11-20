# Pytorch CIFAR10图像分类 MobileNet v1篇

[toc]

## 4.定义网络（MobileNet v1）

在之前的文章中讲的AlexNet、VGG、GoogLeNet以及ResNet网络，它们都是传统卷积神经网络（都是使用的传统卷积层），缺点在于内存需求大、运算量大导致无法在移动设备以及嵌入式设备上运行。而本文要讲的MobileNet网络就是专门为**移动端，嵌入式端**而设计。

我也看了论文，如果想仔细研究一下MobileNet的话，可以看我的另一篇博客[【论文泛读】轻量化之MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://blog.csdn.net/weixin_45508265/article/details/121766611)

MobileNet网络是由google团队在2017年提出的，专注于移动端或者嵌入式设备中的轻量级CNN网络。相比传统卷积神经网络，在准确率小幅降低的前提下大大减少模型参数与运算量。(相比VGG16准确率减少了0.9%，但模型参数只有VGG的1/32)。

要说MobileNet网络的优点，无疑是其中的**Depthwise Convolution结构**(大大减少运算量和参数数量)。下图展示了传统卷积与DW卷积的差异，在传统卷积中，每个卷积核的channel与输入特征矩阵的channel相等（每个卷积核都会与输入特征矩阵的每一个维度进行卷积运算）。而在DW卷积中，每个卷积核的channel都是等于1的（每个卷积核只负责输入特征矩阵的一个channel，故卷积核的个数必须等于输入特征矩阵的channel数，从而使得输出特征矩阵的channel数也等于输入特征矩阵的channel数）

<img src="https://img-blog.csdnimg.cn/20200426162556656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" />


刚刚说了使用DW卷积后输出特征矩阵的channel是与输入特征矩阵的channel相等的，如果想改变/自定义输出特征矩阵的channel，那只需要在DW卷积后接上一个PW卷积即可，如下图所示，其实PW卷积就是普通的卷积而已（只不过卷积核大小为1）。通常DW卷积和PW卷积是放在一起使用的，一起叫做**Depthwise Separable Convolution（深度可分卷积）**。


<img src="https://img-blog.csdnimg.cn/20200426163946303.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" />


那Depthwise Separable Convolution（深度可分卷积）与传统的卷积相比有到底能节省多少计算量呢，下图对比了这两个卷积方式的计算量，其中Df是输入特征矩阵的宽高（这里假设宽和高相等），Dk是卷积核的大小，M是输入特征矩阵的channel，N是输出特征矩阵的channel，卷积计算量近似等于卷积核的高 x 卷积核的宽 x 卷积核的channel x 输入特征矩阵的高 x 输入特征矩阵的宽（这里假设stride等于1），在我们mobilenet网络中DW卷积都是是使用3x3大小的卷积核。所以理论上普通卷积计算量是DW+PW卷积的8到9倍（公式来源于原论文）：


<img src="https://img-blog.csdnimg.cn/20200426164254206.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" />


在了解完Depthwise Separable Convolution（深度可分卷积）后在看下mobilenet v1的网络结构，左侧的表格是mobileNetv1的网络结构，表中标Conv的表示普通卷积，Conv dw代表刚刚说的DW卷积，s表示步距，根据表格信息就能很容易的搭建出mobileNet v1网络。在mobilenetv1原论文中，还提出了两个超参数，一个是α一个是β。α参数是一个倍率因子，用来调整卷积核的个数，β是控制输入网络的图像尺寸参数，下图右侧给出了使用不同α和β网络的分类准确率，计算量以及模型参数：

<img src="https://img-blog.csdnimg.cn/20200426165207571.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" />



首先我们还是得判断是否可以利用GPU，因为GPU的速度可能会比我们用CPU的速度快20-50倍左右，特别是对卷积神经网络来说，更是提升特别明显。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

接着我们可以定义网络，在pytorch之中，定义我们的深度可分离卷积来说，我们需要调一个groups参数，就可以构建深度可分离卷积了。

```python
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self,in_channels,out_channels,stride=1):
        super(Block,self).__init__()
        # groups参数就是深度可分离卷积的关键
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,
                               padding=1,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
        
# 深度可分离卷积 DepthWise Separable Convolution
class MobileNetV1(nn.Module):
    # (128,2) means conv channel=128, conv stride=2, by default conv stride=1
    cfg = [64,(128,2),128,(256,2),256,(512,2),512,512,512,512,512,(1024,2),1024]
    
    def __init__(self, num_classes=10,alpha=1.0,beta=1.0):
        super(MobileNetV1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.avg = nn.AvgPool2d(kernel_size=2)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(1024,num_classes)
    
    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x,int) else x[0]
            stride = 1 if isinstance(x,int) else x[1]
            layers.append(Block(in_channels,out_channels,stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avg(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x
```

```python
net = MobileNetV1(num_classes=10).to(device)
summary(net,(2,3,32,32))
```

> ```python
> ==========================================================================================
> Layer (type:depth-idx)                   Output Shape              Param #
> ==========================================================================================
> MobileNetV1                              --                        --
> ├─Sequential: 1-1                        [2, 32, 30, 30]           --
> │    └─Conv2d: 2-1                       [2, 32, 30, 30]           864
> │    └─BatchNorm2d: 2-2                  [2, 32, 30, 30]           64
> │    └─ReLU: 2-3                         [2, 32, 30, 30]           --
> ├─Sequential: 1-2                        [2, 1024, 2, 2]           --
> │    └─Block: 2-4                        [2, 64, 30, 30]           --
> │    │    └─Conv2d: 3-1                  [2, 32, 30, 30]           288
> │    │    └─BatchNorm2d: 3-2             [2, 32, 30, 30]           64
> │    │    └─ReLU: 3-3                    [2, 32, 30, 30]           --
> │    │    └─Conv2d: 3-4                  [2, 64, 30, 30]           2,048
> │    │    └─BatchNorm2d: 3-5             [2, 64, 30, 30]           128
> │    │    └─ReLU: 3-6                    [2, 64, 30, 30]           --
> │    └─Block: 2-5                        [2, 128, 15, 15]          --
> │    │    └─Conv2d: 3-7                  [2, 64, 15, 15]           576
> │    │    └─BatchNorm2d: 3-8             [2, 64, 15, 15]           128
> │    │    └─ReLU: 3-9                    [2, 64, 15, 15]           --
> │    │    └─Conv2d: 3-10                 [2, 128, 15, 15]          8,192
> │    │    └─BatchNorm2d: 3-11            [2, 128, 15, 15]          256
> │    │    └─ReLU: 3-12                   [2, 128, 15, 15]          --
> │    └─Block: 2-6                        [2, 128, 15, 15]          --
> │    │    └─Conv2d: 3-13                 [2, 128, 15, 15]          1,152
> │    │    └─BatchNorm2d: 3-14            [2, 128, 15, 15]          256
> │    │    └─ReLU: 3-15                   [2, 128, 15, 15]          --
> │    │    └─Conv2d: 3-16                 [2, 128, 15, 15]          16,384
> │    │    └─BatchNorm2d: 3-17            [2, 128, 15, 15]          256
> │    │    └─ReLU: 3-18                   [2, 128, 15, 15]          --
> │    └─Block: 2-7                        [2, 256, 8, 8]            --
> │    │    └─Conv2d: 3-19                 [2, 128, 8, 8]            1,152
> │    │    └─BatchNorm2d: 3-20            [2, 128, 8, 8]            256
> │    │    └─ReLU: 3-21                   [2, 128, 8, 8]            --
> │    │    └─Conv2d: 3-22                 [2, 256, 8, 8]            32,768
> │    │    └─BatchNorm2d: 3-23            [2, 256, 8, 8]            512
> │    │    └─ReLU: 3-24                   [2, 256, 8, 8]            --
> │    └─Block: 2-8                        [2, 256, 8, 8]            --
> │    │    └─Conv2d: 3-25                 [2, 256, 8, 8]            2,304
> │    │    └─BatchNorm2d: 3-26            [2, 256, 8, 8]            512
> │    │    └─ReLU: 3-27                   [2, 256, 8, 8]            --
> │    │    └─Conv2d: 3-28                 [2, 256, 8, 8]            65,536
> │    │    └─BatchNorm2d: 3-29            [2, 256, 8, 8]            512
> │    │    └─ReLU: 3-30                   [2, 256, 8, 8]            --
> │    └─Block: 2-9                        [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-31                 [2, 256, 4, 4]            2,304
> │    │    └─BatchNorm2d: 3-32            [2, 256, 4, 4]            512
> │    │    └─ReLU: 3-33                   [2, 256, 4, 4]            --
> │    │    └─Conv2d: 3-34                 [2, 512, 4, 4]            131,072
> │    │    └─BatchNorm2d: 3-35            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-36                   [2, 512, 4, 4]            --
> │    └─Block: 2-10                       [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-37                 [2, 512, 4, 4]            4,608
> │    │    └─BatchNorm2d: 3-38            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-39                   [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-40                 [2, 512, 4, 4]            262,144
> │    │    └─BatchNorm2d: 3-41            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-42                   [2, 512, 4, 4]            --
> │    └─Block: 2-11                       [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-43                 [2, 512, 4, 4]            4,608
> │    │    └─BatchNorm2d: 3-44            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-45                   [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-46                 [2, 512, 4, 4]            262,144
> │    │    └─BatchNorm2d: 3-47            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-48                   [2, 512, 4, 4]            --
> │    └─Block: 2-12                       [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-49                 [2, 512, 4, 4]            4,608
> │    │    └─BatchNorm2d: 3-50            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-51                   [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-52                 [2, 512, 4, 4]            262,144
> │    │    └─BatchNorm2d: 3-53            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-54                   [2, 512, 4, 4]            --
> │    └─Block: 2-13                       [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-55                 [2, 512, 4, 4]            4,608
> │    │    └─BatchNorm2d: 3-56            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-57                   [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-58                 [2, 512, 4, 4]            262,144
> │    │    └─BatchNorm2d: 3-59            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-60                   [2, 512, 4, 4]            --
> │    └─Block: 2-14                       [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-61                 [2, 512, 4, 4]            4,608
> │    │    └─BatchNorm2d: 3-62            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-63                   [2, 512, 4, 4]            --
> │    │    └─Conv2d: 3-64                 [2, 512, 4, 4]            262,144
> │    │    └─BatchNorm2d: 3-65            [2, 512, 4, 4]            1,024
> │    │    └─ReLU: 3-66                   [2, 512, 4, 4]            --
> │    └─Block: 2-15                       [2, 1024, 2, 2]           --
> │    │    └─Conv2d: 3-67                 [2, 512, 2, 2]            4,608
> │    │    └─BatchNorm2d: 3-68            [2, 512, 2, 2]            1,024
> │    │    └─ReLU: 3-69                   [2, 512, 2, 2]            --
> │    │    └─Conv2d: 3-70                 [2, 1024, 2, 2]           524,288
> │    │    └─BatchNorm2d: 3-71            [2, 1024, 2, 2]           2,048
> │    │    └─ReLU: 3-72                   [2, 1024, 2, 2]           --
> │    └─Block: 2-16                       [2, 1024, 2, 2]           --
> │    │    └─Conv2d: 3-73                 [2, 1024, 2, 2]           9,216
> │    │    └─BatchNorm2d: 3-74            [2, 1024, 2, 2]           2,048
> │    │    └─ReLU: 3-75                   [2, 1024, 2, 2]           --
> │    │    └─Conv2d: 3-76                 [2, 1024, 2, 2]           1,048,576
> │    │    └─BatchNorm2d: 3-77            [2, 1024, 2, 2]           2,048
> │    │    └─ReLU: 3-78                   [2, 1024, 2, 2]           --
> ├─AvgPool2d: 1-3                         [2, 1024, 1, 1]           --
> ├─Linear: 1-4                            [2, 10]                   10,250
> ==========================================================================================
> Total params: 3,217,226
> Trainable params: 3,217,226
> Non-trainable params: 0
> Total mult-adds (M): 90.33
> ==========================================================================================
> Input size (MB): 0.02
> Forward/backward pass size (MB): 12.22
> Params size (MB): 12.87
> Estimated Total Size (MB): 25.11
> ==========================================================================================
> ```

首先从我们summary可以看到，我们定义模型的参数量和计算量都是比较小的，这也是作为MobileNet的一大有点，我们输入的是（batch，3，32，32）的张量，并且这里也能看到每一层后我们的图像输出大小的变化，最后输出10个参数，再通过softmax函数就可以得到我们每个类别的概率了。

我们也可以打印出我们的模型观察一下

```python
MobileNetV1(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (avg): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (layers): Sequential(
    (0): Block(
      (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (1): Block(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (2): Block(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (3): Block(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (4): Block(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (5): Block(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (6): Block(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (7): Block(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (8): Block(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (9): Block(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (10): Block(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (11): Block(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
    (12): Block(
      (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
      (bn1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
      (conv2): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU()
    )
  )
  (linear): Linear(in_features=1024, out_features=10, bias=True)
)
```

如果你的电脑有多个GPU，这段代码可以利用GPU进行并行计算，加快运算速度

```python
net = MobileNetV1(num_classes=10).to(device)
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
save_path = './model/MoblieNetV1.pth'
```

我定义了一个train函数，在train函数中进行一个训练，并保存我们训练后的模型

```python
from utils import plot_history
from utils import train
Acc, Loss, Lr = train(net, trainloader, testloader, epoch, optimizer, criterion, scheduler, save_path, verbose = True)
```

> ```python
> Epoch [  1/ 20]  Train Loss:1.755525  Train Acc:35.18% Test Loss:1.482050  Test Acc:45.63%  Learning Rate:0.100000	Time 01:02
> Epoch [  2/ 20]  Train Loss:1.359267  Train Acc:50.75% Test Loss:1.313756  Test Acc:53.10%  Learning Rate:0.100000	Time 00:58
> Epoch [  3/ 20]  Train Loss:1.123205  Train Acc:60.09% Test Loss:1.083381  Test Acc:61.47%  Learning Rate:0.100000	Time 00:57
> Epoch [  4/ 20]  Train Loss:0.946333  Train Acc:67.18% Test Loss:0.966875  Test Acc:66.20%  Learning Rate:0.100000	Time 00:57
> Epoch [  5/ 20]  Train Loss:0.842723  Train Acc:70.96% Test Loss:0.965495  Test Acc:67.11%  Learning Rate:0.100000	Time 00:55
> Epoch [  6/ 20]  Train Loss:0.781058  Train Acc:73.30% Test Loss:0.895774  Test Acc:69.28%  Learning Rate:0.100000	Time 00:54
> Epoch [  7/ 20]  Train Loss:0.748482  Train Acc:74.32% Test Loss:0.839037  Test Acc:71.43%  Learning Rate:0.100000	Time 00:57
> Epoch [  8/ 20]  Train Loss:0.727354  Train Acc:75.25% Test Loss:0.762057  Test Acc:74.48%  Learning Rate:0.100000	Time 01:00
> Epoch [  9/ 20]  Train Loss:0.701578  Train Acc:76.08% Test Loss:0.817960  Test Acc:72.46%  Learning Rate:0.100000	Time 00:57
> Epoch [ 10/ 20]  Train Loss:0.690966  Train Acc:76.36% Test Loss:0.908296  Test Acc:68.63%  Learning Rate:0.100000	Time 00:58
> Epoch [ 11/ 20]  Train Loss:0.680218  Train Acc:76.78% Test Loss:0.778566  Test Acc:73.70%  Learning Rate:0.100000	Time 00:57
> Epoch [ 12/ 20]  Train Loss:0.673814  Train Acc:77.26% Test Loss:0.890464  Test Acc:71.39%  Learning Rate:0.100000	Time 00:58
> Epoch [ 13/ 20]  Train Loss:0.673683  Train Acc:77.30% Test Loss:0.792788  Test Acc:72.98%  Learning Rate:0.100000	Time 00:58
> Epoch [ 14/ 20]  Train Loss:0.663810  Train Acc:77.39% Test Loss:0.884162  Test Acc:70.07%  Learning Rate:0.100000	Time 00:59
> Epoch [ 15/ 20]  Train Loss:0.656750  Train Acc:77.84% Test Loss:0.757378  Test Acc:74.75%  Learning Rate:0.100000	Time 01:00
> Epoch [ 16/ 20]  Train Loss:0.648556  Train Acc:78.11% Test Loss:0.954492  Test Acc:67.81%  Learning Rate:0.100000	Time 01:00
> Epoch [ 17/ 20]  Train Loss:0.646635  Train Acc:78.01% Test Loss:0.826150  Test Acc:71.81%  Learning Rate:0.100000	Time 01:03
> Epoch [ 18/ 20]  Train Loss:0.638446  Train Acc:78.42% Test Loss:0.741989  Test Acc:74.34%  Learning Rate:0.100000	Time 01:01
> Epoch [ 19/ 20]  Train Loss:0.634496  Train Acc:78.42% Test Loss:0.817648  Test Acc:73.02%  Learning Rate:0.100000	Time 00:59
> Epoch [ 20/ 20]  Train Loss:0.631237  Train Acc:78.66% Test Loss:0.885431  Test Acc:69.61%  Learning Rate:0.100000	Time 01:01
> ```

接着可以分别打印，损失函数曲线，准确率曲线和学习率曲线

```python
plot_history(epoch ,Acc, Loss, Lr)
```

### 损失函数曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/03e9c582b9b844caad34d52b9bcec809.png)

### 准确率曲线



![在这里插入图片描述](https://img-blog.csdnimg.cn/f2806f5024fd4bebbecc97b623e073b5.png)

### 学习率曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/63417196174a454da5d1e35285fe78e3.png)

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
> Accuracy of the network on the 10000 test images: 69.23 %
> ```

可以看到MobileNet的模型在测试集中准确率达到70%左右

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
> Accuracy of airplane : 63.50 %
> Accuracy of automobile : 79.20 %
> Accuracy of  bird : 50.30 %
> Accuracy of   cat : 56.30 %
> Accuracy of  deer : 49.50 %
> Accuracy of   dog : 76.20 %
> Accuracy of  frog : 91.20 %
> Accuracy of horse : 71.60 %
> Accuracy of  ship : 90.50 %
> Accuracy of truck : 69.40 %
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
> Accuracy Rate = 71.875%
> <Figure size 1800x288 with 0 Axes>
> ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/30a589e0507a486d955ca9cf3ea84efd.png)

## 8. 保存模型

```python
torch.save(net,save_path[:-4]+'_'+str(epoch)+'.pth')
# torch.save(net, './model/MobileNetv1.pth')
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

model = MobileNetV1(num_classes=10)

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
> 概率 tensor([[9.9107e-01, 2.8922e-05, 1.3522e-03, 2.8523e-04, 3.5258e-04, 2.8689e-05,
>             2.4901e-05, 9.8299e-05, 6.4787e-03, 2.8210e-04]], device='cuda:0',
>           grad_fn=<SoftmaxBackward>)
> 类别 0
> tensor([7.6616], device='cuda:0')
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
> 概率 tensor([[0.0743, 0.0026, 0.0117, 0.6632, 0.0266, 0.1934, 0.0139, 0.0053, 0.0061,
>             0.0030]], device='cuda:0', grad_fn=<SoftmaxBackward>)
> 类别 3
> tensor([3.5014], device='cuda:0')
> 分类 cat
> ```

可以看到，分类都是正确的，置信度有66.32%，而且计算速度很快，很不错，这也是MobileNet的优点。



## 10.总结

在这个模型中，其实我们只迭代了20次，MobileNet相对于那些臃肿的VGG或者ResNet模型来说，他的计算速度是可以达到ms级别的，非常的少，谷歌是针对嵌入式设备进行开发的，我们也可以从MobileNet这个Mobile中看得出来，在我们推理的时候，他都是非常轻量化的模型，很不错的。

在之后，谷歌又提出了MobileNetv2和v3，不仅比v1速度更快，准确率也更高，之后也会进行讲解。



顺带提一句，我们的数据和代码都在我的汇总篇里有说明，如果需要，可以自取

这里再贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)



