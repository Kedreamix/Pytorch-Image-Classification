# Pytorch CIFAR10图像分类 MobileNet v2篇

[toc]

再次介绍一下我的专栏，很适合大家初入深度学习或者是Pytorch和Keras，希望这能够帮助初学深度学习的同学一个入门Pytorch或者Keras的项目和在这之中更加了解Pytorch&Keras和各个图像分类的模型。

他有比较清晰的可视化结构和架构，除此之外，我是用jupyter写的，所以说在文章整体架构可以说是非常清晰，可以帮助你快速学习到各个模块的知识，而不是通过python脚本一行一行的看，这样的方式是符合初学者的。

除此之外，如果你需要变成脚本形式，也是很简单的。
这里贴一下汇总篇：[汇总篇](https://redamancy.blog.csdn.net/article/details/127859300)

## 4. 定义网络（MobileNet v2）

在之前的文章中讲的AlexNet、VGG、GoogLeNet以及ResNet网络，它们都是传统卷积神经网络（都是使用的传统卷积层），缺点在于内存需求大、运算量大导致无法在移动设备以及嵌入式设备上运行。而本文要讲的MobileNet网络就是专门为**移动端，嵌入式端**而设计。

### MobileNet v2

在MobileNet v1的网络结构表中能够发现，网络的结构就像VGG一样是个直筒型的，不像ResNet网络有shorcut之类的连接方式。而且有人反映说MobileNet v1网络中的DW卷积很容易训练废掉，效果并没有那么理想。所以我们接着看下MobileNet v2网络。

MobileNet v2网络是由google团队在2018年提出的，**相比MobileNet V1网络，准确率更高，模型更小**。刚刚说了MobileNet v1网络中的亮点是DW卷积，那么在MobileNet v2中的亮点就是**Inverted residual block（倒残差结构）**，如下下图所示，左侧是ResNet网络中的残差结构，右侧就是MobileNet v2中的倒残差结构。**在残差结构中是1x1卷积降维->3x3卷积->1x1卷积升维，在倒残差结构中正好相反，是1x1卷积升维->3x3DW卷积->1x1卷积降维**。为什么要这样做，原文的解释是高维信息通过ReLU激活函数后丢失的信息更少（**注意倒残差结构中基本使用的都是ReLU6激活函数，但是最后一个1x1的卷积层使用的是线性激活函数**）。

<img src="https://img-blog.csdnimg.cn/20200426171005366.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" />

在使用倒残差结构时需要注意下，并不是所有的倒残差结构都有shortcut连接，只有当stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut连接（只有当shape相同时，两个矩阵才能做加法运算，当stride=1时并不能保证输入特征矩阵的channel与输出特征矩阵的channel相同）。


<img src="https://img-blog.csdnimg.cn/20200426171906852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" />


下图是MobileNet v2网络的结构表，其中t代表的是扩展因子（倒残差结构中第一个1x1卷积的扩展因子），c代表输出特征矩阵的channel，n代表倒残差结构重复的次数，s代表步距（注意：这里的步距只是针对重复n次的第一层倒残差结构，后面的都默认为1）。

<img src="https://img-blog.csdnimg.cn/20200426172803520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" />



首先我们还是得判断是否可以利用GPU，因为GPU的速度可能会比我们用CPU的速度快20-50倍左右，特别是对卷积神经网络来说，更是提升特别明显。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

接着我们可以定义网络，在pytorch之中，定义我们的深度可分离卷积来说，我们需要调一个groups参数，就可以构建深度可分离卷积了

```python
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

```

定义了基本模块后，我们就构建我们的倒残差结构和我们主要的MobileNetv2的结构

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

```python
net = MobileNetV2(num_classes=10).to(device)
```

### summary查看网络

我们可以通过summary来看到，模型的维度的变化，这个也是和论文是匹配的，经过层后shape的变化，是否最后也是输出(batch,shape)

```python
summary(net,(2,3,32,32))
```



> ```python
> ===============================================================================================
> Layer (type:depth-idx)                        Output Shape              Param #
> ===============================================================================================
> MobileNetV2                                   [2, 10]                   --
> ├─Sequential: 1-1                             [2, 1280, 1, 1]           --
> │    └─ConvBNReLU: 2-1                        [2, 32, 16, 16]           --
> │    │    └─Conv2d: 3-1                       [2, 32, 16, 16]           864
> │    │    └─BatchNorm2d: 3-2                  [2, 32, 16, 16]           64
> │    │    └─ReLU6: 3-3                        [2, 32, 16, 16]           --
> │    └─InvertedResidual: 2-2                  [2, 16, 16, 16]           --
> │    │    └─Sequential: 3-4                   [2, 16, 16, 16]           896
> │    └─InvertedResidual: 2-3                  [2, 24, 8, 8]             --
> │    │    └─Sequential: 3-5                   [2, 24, 8, 8]             5,136
> │    └─InvertedResidual: 2-4                  [2, 24, 8, 8]             --
> │    │    └─Sequential: 3-6                   [2, 24, 8, 8]             8,832
> │    └─InvertedResidual: 2-5                  [2, 32, 4, 4]             --
> │    │    └─Sequential: 3-7                   [2, 32, 4, 4]             10,000
> │    └─InvertedResidual: 2-6                  [2, 32, 4, 4]             --
> │    │    └─Sequential: 3-8                   [2, 32, 4, 4]             14,848
> │    └─InvertedResidual: 2-7                  [2, 32, 4, 4]             --
> │    │    └─Sequential: 3-9                   [2, 32, 4, 4]             14,848
> │    └─InvertedResidual: 2-8                  [2, 64, 2, 2]             --
> │    │    └─Sequential: 3-10                  [2, 64, 2, 2]             21,056
> │    └─InvertedResidual: 2-9                  [2, 64, 2, 2]             --
> │    │    └─Sequential: 3-11                  [2, 64, 2, 2]             54,272
> │    └─InvertedResidual: 2-10                 [2, 64, 2, 2]             --
> │    │    └─Sequential: 3-12                  [2, 64, 2, 2]             54,272
> │    └─InvertedResidual: 2-11                 [2, 64, 2, 2]             --
> │    │    └─Sequential: 3-13                  [2, 64, 2, 2]             54,272
> │    └─InvertedResidual: 2-12                 [2, 96, 2, 2]             --
> │    │    └─Sequential: 3-14                  [2, 96, 2, 2]             66,624
> │    └─InvertedResidual: 2-13                 [2, 96, 2, 2]             --
> │    │    └─Sequential: 3-15                  [2, 96, 2, 2]             118,272
> │    └─InvertedResidual: 2-14                 [2, 96, 2, 2]             --
> │    │    └─Sequential: 3-16                  [2, 96, 2, 2]             118,272
> │    └─InvertedResidual: 2-15                 [2, 160, 1, 1]            --
> │    │    └─Sequential: 3-17                  [2, 160, 1, 1]            155,264
> │    └─InvertedResidual: 2-16                 [2, 160, 1, 1]            --
> │    │    └─Sequential: 3-18                  [2, 160, 1, 1]            320,000
> │    └─InvertedResidual: 2-17                 [2, 160, 1, 1]            --
> │    │    └─Sequential: 3-19                  [2, 160, 1, 1]            320,000
> │    └─InvertedResidual: 2-18                 [2, 320, 1, 1]            --
> │    │    └─Sequential: 3-20                  [2, 320, 1, 1]            473,920
> │    └─ConvBNReLU: 2-19                       [2, 1280, 1, 1]           --
> │    │    └─Conv2d: 3-21                      [2, 1280, 1, 1]           409,600
> │    │    └─BatchNorm2d: 3-22                 [2, 1280, 1, 1]           2,560
> │    │    └─ReLU6: 3-23                       [2, 1280, 1, 1]           --
> ├─AdaptiveAvgPool2d: 1-2                      [2, 1280, 1, 1]           --
> ├─Sequential: 1-3                             [2, 10]                   --
> │    └─Dropout: 2-20                          [2, 1280]                 --
> │    └─Linear: 2-21                           [2, 10]                   12,810
> ===============================================================================================
> Total params: 2,236,682
> Trainable params: 2,236,682
> Non-trainable params: 0
> Total mult-adds (M): 12.32
> ===============================================================================================
> Input size (MB): 0.02
> Forward/backward pass size (MB): 4.36
> Params size (MB): 8.95
> Estimated Total Size (MB): 13.33
> ===============================================================================================
> ```

首先从我们summary可以看到，我们定义模型的参数量和计算量都是比较小的，这也是作为MobileNet的一大特点，我们输入的是（batch，3，32，32）的张量，并且这里也能看到每一层后我们的图像输出大小的变化，最后输出10个参数，再通过softmax函数就可以得到我们每个类别的概率了。

我们也可以打印出我们的模型观察一下

```python
MobileNetV2(
  (features): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (9): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (16): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (17): InvertedResidual(
      (conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (18): ConvBNReLU(
      (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=10, bias=True)
  )
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
net = MobileNetV2(num_classes=10)
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
save_path = './model/MoblieNetv2.pth'
```

这次更新了tensorboard的可视化，可以得到更好看的图片，并且能可视化出不错的结果

```python
# 使用tensorboard
from torch.utils.tensorboard import SummaryWriter
os.makedirs("./logs", exist_ok=True)
tbwriter = SummaryWriter(log_dir='./logs/MoblieNetv2', comment='MoblieNetV2')  # 使用tensorboard记录中间输出
tbwriter.add_graph(model= net, input_to_model=torch.randn(size=(2, 3, 32, 32)))
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
> Train Epoch 1/10: 100%|██████████| 196/196 [00:22<00:00,  8.76it/s, Train Acc=0.215, Train Loss=2.53]
> Test Epoch 1/10: 100%|██████████| 40/40 [00:01<00:00, 25.69it/s, Test Acc=0.232, Test Loss=2.08]
> Train Epoch 2/10: 100%|██████████| 196/196 [00:22<00:00,  8.76it/s, Train Acc=0.235, Train Loss=2.22]
> Test Epoch 2/10: 100%|██████████| 40/40 [00:01<00:00, 26.88it/s, Test Acc=0.247, Test Loss=2.2] 
> Train Epoch 3/10: 100%|██████████| 196/196 [00:22<00:00,  8.86it/s, Train Acc=0.295, Train Loss=1.97]
> Test Epoch 3/10: 100%|██████████| 40/40 [00:01<00:00, 24.49it/s, Test Acc=0.113, Test Loss=12.1]
> Epoch [  3/ 10]  Train Loss:1.974274  Train Acc:29.52% Test Loss:12.094494  Test Acc:11.33%  Learning Rate:0.100000
> Train Epoch 4/10: 100%|██████████| 196/196 [00:22<00:00,  8.84it/s, Train Acc=0.304, Train Loss=1.91]
> Test Epoch 4/10: 100%|██████████| 40/40 [00:01<00:00, 25.33it/s, Test Acc=0.313, Test Loss=1.79]
> Epoch [  4/ 10]  Train Loss:1.911137  Train Acc:30.38% Test Loss:1.793694  Test Acc:31.35%  Learning Rate:0.100000
> Train Epoch 5/10: 100%|██████████| 196/196 [00:22<00:00,  8.62it/s, Train Acc=0.383, Train Loss=1.65]
> Test Epoch 5/10: 100%|██████████| 40/40 [00:01<00:00, 26.96it/s, Test Acc=0.424, Test Loss=1.58]
> Epoch [  5/ 10]  Train Loss:1.651606  Train Acc:38.30% Test Loss:1.580125  Test Acc:42.42%  Learning Rate:0.100000
> Train Epoch 6/10: 100%|██████████| 196/196 [00:22<00:00,  8.79it/s, Train Acc=0.427, Train Loss=1.54]
> Test Epoch 6/10: 100%|██████████| 40/40 [00:01<00:00, 24.90it/s, Test Acc=0.45, Test Loss=1.49] 
> Epoch [  6/ 10]  Train Loss:1.540308  Train Acc:42.75% Test Loss:1.488657  Test Acc:44.95%  Learning Rate:0.100000
> Train Epoch 7/10: 100%|██████████| 196/196 [00:22<00:00,  8.54it/s, Train Acc=0.48, Train Loss=1.42] 
> Test Epoch 7/10: 100%|██████████| 40/40 [00:01<00:00, 25.39it/s, Test Acc=0.477, Test Loss=1.45]
> Epoch [  7/ 10]  Train Loss:1.421093  Train Acc:47.96% Test Loss:1.445841  Test Acc:47.71%  Learning Rate:0.100000
> Train Epoch 8/10: 100%|██████████| 196/196 [00:21<00:00,  8.94it/s, Train Acc=0.521, Train Loss=1.32]
> Test Epoch 8/10: 100%|██████████| 40/40 [00:01<00:00, 23.63it/s, Test Acc=0.506, Test Loss=1.35]
> Epoch [  8/ 10]  Train Loss:1.320276  Train Acc:52.11% Test Loss:1.349572  Test Acc:50.57%  Learning Rate:0.100000
> Train Epoch 9/10: 100%|██████████| 196/196 [00:22<00:00,  8.72it/s, Train Acc=0.556, Train Loss=1.24]
> Test Epoch 9/10: 100%|██████████| 40/40 [00:01<00:00, 25.30it/s, Test Acc=0.543, Test Loss=1.26]
> Epoch [  9/ 10]  Train Loss:1.235295  Train Acc:55.59% Test Loss:1.259758  Test Acc:54.27%  Learning Rate:0.100000
> Train Epoch 10/10: 100%|██████████| 196/196 [00:22<00:00,  8.54it/s, Train Acc=0.583, Train Loss=1.16]
> Test Epoch 10/10: 100%|██████████| 40/40 [00:01<00:00, 24.25it/s, Test Acc=0.533, Test Loss=1.31]
> Epoch [ 10/ 10]  Train Loss:1.159529  Train Acc:58.34% Test Loss:1.313859  Test Acc:53.32%  Learning Rate:0.100000
> ```

### 训练曲线可视化

接着可以分别打印，损失函数曲线，准确率曲线和学习率曲线

```python
plot_history(epoch ,Acc, Loss, Lr)
```

#### 损失函数曲线

![在这里插入图片描述](https://img-blog.csdnimg.cn/03a4fdcddfe34fc798bb36a360b887ed.png)

#### 准确率曲线



![在这里插入图片描述](https://img-blog.csdnimg.cn/09a4e6cb60a948f58962edfa8a5bc62c.png)

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
> Accuracy of the network on the 10000 test images: 53.70 %
> ```

可以看到MobileNetv2的模型在测试集中准确率达到53.7%左右

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
> Accuracy of airplane : 35.30 %
> Accuracy of automobile : 83.50 %
> Accuracy of  bird : 17.90 %
> Accuracy of   cat : 52.00 %
> Accuracy of  deer : 27.50 %
> Accuracy of   dog : 47.50 %
> Accuracy of  frog : 74.60 %
> Accuracy of horse : 57.70 %
> Accuracy of  ship : 75.10 %
> Accuracy of truck : 65.90 %
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
> Accuracy Rate = 56.640625%
> ```

![在这里插入图片描述](https://img-blog.csdnimg.cn/7a76665ad17243239cd478c36284934f.png)

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
概率 tensor([[9.3877e-01, 4.3625e-03, 1.0835e-02, 6.7719e-04, 2.0643e-03, 1.4232e-04,
         4.9050e-04, 1.6972e-04, 4.0807e-02, 1.6775e-03]], device='cuda:0',
       grad_fn=<SoftmaxBackward>)
类别 0
tensor([5.7688], device='cuda:0')
分类 airplane
```

这里就可以看到，我们最后的结果，分类为plane，我们的置信率大概是93.8%，看起来是很不错的

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
> 概率 tensor([[0.6336, 0.0063, 0.1490, 0.0659, 0.0441, 0.0254, 0.0109, 0.0144, 0.0423,
>             0.0081]], device='cuda:0', grad_fn=<SoftmaxBackward>)
> 类别 0
> tensor([2.9148], device='cuda:0')
> 分类 airplane
> ```

可以看到，分类不正确了，对于cat还是分类到了airplane，这有可能是我们只迭代了10次的原因，我相信继续调参可能能达到70%以上



## 10.总结

在这个模型中，其实我们只迭代了10次，MobileNet相对于那些臃肿的VGG或者ResNet模型来说，他的计算速度是可以达到ms级别的，非常的少，谷歌是针对嵌入式设备进行开发的，我们也可以从MobileNet这个Mobile中看得出来，在我们推理的时候，他都是非常轻量化的模型，很不错的，并且v2比v1,不仅比v1速度更快，准确率也更高，后续的v3之后也会进行讲解。

顺带提一句，我们的数据和代码都在我的汇总篇里有说明，如果需要，可以自取

这里再贴一下汇总篇：[汇总篇](https://redamancy.blog.csdn.net/article/details/127859300)



