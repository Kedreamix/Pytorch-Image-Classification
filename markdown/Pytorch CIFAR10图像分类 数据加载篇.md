# Pytorch CIFAR10图像分类 数据加载与可视化篇

[toc]

这里贴一下汇总篇：[汇总篇](https://blog.csdn.net/weixin_45508265/article/details/119285255)

**Pytorch一般有以下几个流程**

1. 数据读取
2. 数据处理
3. 搭建网络
4. 模型训练
5. 模型上线

这里会先讲一下关于CIFAR10的数据加载和图片可视化，之后的模型篇会对网络进行介绍和实现。

### 1.数据读取

CIFAR-10 是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ arplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。 

与 MNIST 数据集中目比， CIFAR-10 具有以下不同点：

- CIFAR-10 是 3 通道的彩色 RGB 图像，而 MNIST 是灰度图像。
- CIFAR-10 的图片尺寸为 32×32， 而 MNIST 的图片尺寸为 28×28，比 MNIST 稍大。
- 相比于手写字符， CIFAR-10 含有的是现实世界中真实的物体，不仅噪声很大，而且物体的比例、 特征都不尽相同，这为识别带来很大困难。

![在这里插入图片描述](https://img-blog.csdnimg.cn/16f85f24a70e452e8659a1874616420f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70#pic_center)



首先使用`torchvision`加载和归一化我们的训练数据和测试数据。

a、`torchvision`这个东西，实现了常用的一些深度学习的相关的图像数据的加载功能，比如cifar10、Imagenet、Mnist等等的，保存在`torchvision.datasets`模块中。

b、同时，也封装了一些处理数据的方法。保存在`torchvision.transforms`模块中

c、还封装了一些模型和工具封装在相应模型中,比如`torchvision.models`当中就包含了AlexNet，VGG，ResNet，SqueezeNet等模型。



**由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor**

首先定义了一个变换transform，利用的是上面提到的transforms模块中的Compose( )把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换

`transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差，注意通道顺序是 R G B ，用过opencv的同学应该知道openCV读出来的图像是 BRG顺序。这两个tuple数据是用来对RGB 图像做归一化的，如其名称 Normalize 所示这里都取0.5只是一个近似的操作，实际上其均值和方差并不是这么多，但是就这个示例而言 影响可不计。精确值是通过分别计算R,G,B三个通道的数据算出来的。

```python
transform = transforms.Compose([
#     transforms.CenterCrop(224),
    transforms.RandomCrop(32,padding=4), # 数据增广
    transforms.RandomHorizontalFlip(),  # 数据增广
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 
```

 `trainloader`其实是一个比较重要的东西，我们后面就是通过`trainloader`把数据传入网络，当然这里的`trainloader`其实是个变量名，可以随便取，重点是他是由后面的`torch.utils.data.DataLoader()`定义的，这个东西来源于`torch.utils.data`模块

```python
Batch_Size = 256
```

```python
trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
testset = datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size,shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size,shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

> ```python
> Files already downloaded and verified
> Files already downloaded and verified
> ```

### 2. 查看数据（格式，大小，形状）

首先可以查看类别

```python
classes = trainset.classes
classes
```

> ```python
> ['airplane',
>  'automobile',
>  'bird',
>  'cat',
>  'deer',
>  'dog',
>  'frog',
>  'horse',
>  'ship',
>  'truck']
> ```

```python
trainset.class_to_idx
```

> ```python
> {'airplane': 0,
>  'automobile': 1,
>  'bird': 2,
>  'cat': 3,
>  'deer': 4,
>  'dog': 5,
>  'frog': 6,
>  'horse': 7,
>  'ship': 8,
>  'truck': 9}
> ```

也可以查看一下训练集的数据

```python
trainset.data.shape #50000是图片数量，32x32是图片大小，3是通道数量RGB
```

> ```python
> (50000, 32, 32, 3)
> ```

查看数据类型

```python
#查看数据类型
print(type(trainset.data))
print(type(trainset))
```

> ```python
> <class 'numpy.ndarray'>
> <class 'torchvision.datasets.cifar.CIFAR10'>
> ```
>
> 

**总结：**

`trainset.data.shape`是标准的numpy.ndarray类型，其中50000是图片数量，32x32是图片大小，3是通道数量RGB；
`trainset`是标准的？？类型，其中50000为图片数量，0表示取前面的数据，2表示3通道数RGB，32*32表示图片大小

###  3. 查看图片

接下来我们对图片进行可视化

```python
import numpy as np
import matplotlib.pyplot as plt
plt.imshow(trainset.data[0])
im,label = iter(trainloader).next()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/d2733fe9714446caa0f6ff0d8501adcd.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

#### np.ndarray转为torch.Tensor

在深度学习中，原始图像需要转换为深度学习框架自定义的数据格式，在pytorch中，需要转为`torch.Tensor`。
pytorch提供了`torch.Tensor` 与`numpy.ndarray`转换为接口：

| 方法名                  | 作用                            |
| ----------------------- | ------------------------------- |
| `torch.from_numpy(xxx)` | `numpy.ndarray`转为torch.Tensor |
| `tensor1.numpy()`       | 获取tensor1对象的numpy格式数据  |

`torch.Tensor` 高维矩阵的表示： N x C x H x W

`numpy.ndarray` 高维矩阵的表示：N x H x W x C

因此在两者转换的时候需要使用`numpy.transpose( )` 方法 。

```python
def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(),(1,2,0))
    plt.imshow(img)
```

```python
imshow(im[0])
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/bc61a5f1af4b45f696e2a6d8bfc7b223.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

我们也可以批量可视化图片，不过这里需要用到`make_grid`

```python
plt.figure(figsize=(8,12))
imshow(torchvision.utils.make_grid(im[:32]))
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/b9beb6f0fe4b4fcb9ef26f300bbb242b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)