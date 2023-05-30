# ✨Pytorch&Keras CIFAR10图像分类

<!-- TOC -->

- [✨Pytorch&Keras CIFAR10图像分类](#pytorchkeras-cifar10图像分类)
  - [🧑‍🎓适用人群](#‍适用人群)
  - [📚︎博客汇总](#︎博客汇总)
    - [💻︎ Pytorch CIFAR10 图像分类](#︎-pytorch-cifar10-图像分类)
      - [⁉ 数据处理以及工具函数（网络篇前必看）](#⁉-数据处理以及工具函数网络篇前必看)
      - [❕ 网络篇](#-网络篇)
    - [🖥︎ Keras CIFAR10 图像分类](#🖥︎-keras-cifar10-图像分类)
    - [💝 有趣的项目和尝试](#-有趣的项目和尝试)
  - [📅 Comming soon 更新计划](#-comming-soon-更新计划)
  - [🧰 使用方法](#🧰-使用方法)
  - [📚 参考](#-参考)

<!-- /TOC -->
![Keras vs PyTorch vs Caffe - Comparing the Implementation of CNN](https://149695847.v2.pressablecdn.com/wp-content/uploads/2020/08/create-machine-learning-and-deep-learning-models-using-pytorch-and-tensorflow.jpg#pic_center)

## 💪专栏介绍

一开始写这个专栏的初衷是，**有时候有些代码找的太繁琐了，并且找到了还运行不出来，或者运行了，但是没有一个很好的理解**，所以基于此，我写了这个CIFAR10图像分类的专栏，借此希望，大家都能通过我这个专栏，找到自己想要的模型或者学习深度学习的代码。

由于现在深度学习也在热度中，很多时候我们难免需要遇见深度学习的时候，**在一些课程和项目的要求下，我们会发现，一个好的代码和一个好的可视化和清晰解释是能够节省特别特别多的时间的**，基于此，我写出了这个专栏，这个专栏下的所有项目，都是**可运行无差错的。**如果遇见了问题，也可以留言跟我沟通

---

## 🧑‍🎓适用人群

他很适合大家**初入深度学习或者是Pytorch和Keras**，希望这能够帮助初学深度学习的同学一个入门Pytorch或者Keras的项目和在这之中更加了解Pytorch&Keras和各个图像分类的模型。

他有比较清晰的可视化结构和架构，除此之外，我是**用jupyter写的，所以说在文章整体架构可以说是非常清晰**，可以帮助你快速学习到**各个模块的知识**，而不是通过python脚本一行一行的看，这样的方式是符合初学者的。

除此之外，如果你需要变成脚本形式，也是很简单的。

---

## 📚︎博客汇总

为了使得文章体系结构更加清晰，这里给出**Pytorch&Keras对于CIFAR10图像分类的所有资料汇总**，也就是我的博客汇总，也告诉大家，我做了什么工作，如果大家有兴趣订阅我的专栏亦或者是有什么其他模型的想法，也可以评论留言，我也可以进行去学习的。

这部分也方便大家看到介绍，并且快速找到自己所需要的代码进行学习和了解

---

### 💻︎ Pytorch CIFAR10 图像分类

在看网络篇的时候，可以先看前三个，前三个说明的如何加载数据包括数据的预处理以及进行可视化，工具函数篇介绍了如何构建训练的函数，并且有时候会遇到一部分utils.py的错误，在这里面都有很好的解释和进行学习。

#### ⁉ 数据处理以及工具函数（网络篇前必看）

- [Pytorch CIFAR10图像分类 数据加载与可视化篇](https://blog.csdn.net/weixin_45508265/article/details/119285113)
- [Pytorch CIFAR10图像分类 工具函数utils篇](https://redamancy.blog.csdn.net/article/details/121589217) 
- [Pytorch CIFAR10图像分类 工具函数utils更新v2.0篇](https://redamancy.blog.csdn.net/article/details/127856569)

#### ❕ 网络篇

- [Pytorch CIFAR10图像分类 自定义网络篇](https://blog.csdn.net/weixin_45508265/article/details/119305277)
- [Pytorch CIFAR10图像分类 LeNet5篇](https://blog.csdn.net/weixin_45508265/article/details/119305673)
- [Pytorch CIFAR10图像分类 AlexNet篇](https://blog.csdn.net/weixin_45508265/article/details/119305848)  
- [Pytorch CIFAR10图像分类 VGG篇](https://blog.csdn.net/weixin_45508265/article/details/119332904) 
- [Pytorch CIFAR10图像分类 GoogLeNet篇](https://blog.csdn.net/weixin_45508265/article/details/119399239)
- [Pytorch CIFAR10图像分类 ResNet篇](https://blog.csdn.net/weixin_45508265/article/details/119532143) 
- [Pytorch CIFAR10图像分类 DenseNet篇](https://blog.csdn.net/weixin_45508265/article/details/119648036)
- [Pytorch CIFAR10图像分类 MobieNetv1篇](https://redamancy.blog.csdn.net/article/details/124636103)
- [Pytorch CIFAR10图像分类 MobileNetv2篇](https://redamancy.blog.csdn.net/article/details/127946431) 
- [Pytorch CIFAR10图像分类 ResNeXt篇](https://redamancy.blog.csdn.net/article/details/126655797)  
- [Pytorch CIFAR10图像分类 ZFNet篇](https://blog.csdn.net/weixin_45508265/article/details/128560595)
- [Pytorch CIFAR10图像分类 SENet篇](https://blog.csdn.net/weixin_45508265/article/details/130938341)
- [Pytorch CIFAR10 图像分类 Vision Transformer篇](https://redamancy.blog.csdn.net/article/details/126751948)
- [Pytorch CIFAR10图像分类 EfficientNet篇](https://blog.csdn.net/weixin_45508265/article/details/128585354)
- [Pytorch CIFAR10图像分类 ShuffleNet篇](https://blog.csdn.net/weixin_45508265/article/details/130945031)

> 具体的详情可以关注[Pytorch CIFAR10图像分类汇总篇](https://redamancy.blog.csdn.net/article/details/119285255)

---

### 🖥︎ Keras CIFAR10 图像分类

- [Keras CIFAR-10分类 SVM 分类器篇][1]
- [Keras CIFAR-10分类 LeNet-5篇][2]
- [Keras CIFAR-10分类 AlexNet篇][3]
- [Keras CIFAR-10分类 GoogleNet篇][4]
- [Keras CIFAR-10分类 VGG篇][5]
- [Keras CIFAR-10分类 ResNet篇][6]
- [Keras CIFAR-10分类 DenseNet篇][7]

> 具体的详情可以关注[Keras CIFAR-10 分类汇总篇](https://blog.csdn.net/weixin_45508265/article/details/127859003)

---

### 💝 有趣的项目和尝试

- [MAE实现及预训练可视化 （CIFAR-Pytorch）][MAE]

---

## 📅 Comming soon 更新计划

- [x] LetNet
- [x] AlexNet
- [x] VGG
- [x] ResNet
- [x] GoogLeNet
- [x] DenseNet
- [x] ResNeXt
- [x] MobileNetv1
- [x] MobileNetv2
- [x] ZFNet
- [x] SeNet
- [x] Efficientent
- [x] ViT
- [x] ShuffleNetv1
- [ ] ShuffleNetv2
- [ ] Swin-Transformer
- [ ] ConvNeXt
- [ ] ConvNeXtv2

---

## 🧰 使用方法

下载`CIFAR10`里所有文件，直接运行ipynb即可，由于我是利用一个工具函数进行训练的，所以**切记utils.py是必不可少的。**

运行ipynb文件即可，对于网络的py文件会持续更新，之后会利用一个函数来选取对应的网络进行训练得到结果。

---

## 📚 参考

除此之外，我还为图像分类这个专栏录了一下我的视频讲解，感兴趣的小伙伴可以来我的B站看视频进行学习，啃代码的时候，可能听一下也会有更多的感触哦
[https://space.bilibili.com/241286257/channel/seriesdetail?sid=2075039](https://space.bilibili.com/241286257/channel/seriesdetail?sid=2075039)

---

最后这个是我写的一个pytorch的基础的介绍，[Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)，从0开始的完整的介绍pytorch和pytorch的简单语法，并且里面有一些项目和学习，还是很不错的哦，可以查看，除此之外，有什么想法可以加我wx: `pikachu2biubiu`聊哦，需要什么帮助也可以付费聊咨询。

![二维码](C:\Users\86137\Documents\GitHub\Pytorch-Image-Classification\QR.png)











[1]: https://redamancy.blog.csdn.net/article/details/126445778
[2]: https://redamancy.blog.csdn.net/article/details/126446810
[3]: https://redamancy.blog.csdn.net/article/details/126590621
[4]: https://redamancy.blog.csdn.net/article/details/126591761
[5]: https://redamancy.blog.csdn.net/article/details/126669709
[6]: https://redamancy.blog.csdn.net/article/details/127827641
[7]: https://redamancy.blog.csdn.net/article/details/127828318
[MAE]: https://redamancy.blog.csdn.net/article/details/126863995

