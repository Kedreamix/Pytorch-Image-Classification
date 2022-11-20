# Pytorch  CIFAR10 图像分类篇 汇总



![在这里插入图片描述](https://img-blog.csdnimg.cn/f37d635b66334021a12521d6bed88d87.png#pic_center)

接下来我会分别利用深度学习的方法，用Pytorch实现我们的CIFAR10的图像分类

大概预计的模型有LeNet，AlexNet，VGG，GoogLeNet，ResNet，DenseNet，MobileNet，Vision Transformer， ResNeXt等，除此之外也会陆续补充

希望这能够帮助初学机器学习的同学一个入门Pytorch的项目和在这之中更加了解Pytorch和各个图像分类的模型。

- [Pytorch CIFAR10图像分类 数据加载与可视化篇](https://blog.csdn.net/weixin_45508265/article/details/119285113)   [B站视频](https://www.bilibili.com/video/BV1FP4y1g7sc) 
- [Pytorch CIFAR10图像分类 工具函数utils篇](https://redamancy.blog.csdn.net/article/details/121589217) [Online Demo](https://drive.google.com/file/d/1HESbXuEb__9eXqq4tAl8owsb1FKhpO2i/view?usp=sharing)
- [Pytorch CIFAR10图像分类 自定义网络篇](https://blog.csdn.net/weixin_45508265/article/details/119305277)  [B站视频](https://www.bilibili.com/video/BV1MF41147gZ) [Colab Demo](https://colab.research.google.com/drive/1BO0wSY3w3xma-oATLyIQRq19qjX1FxF7?usp=sharing) for 自定义网络 [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1BO0wSY3w3xma-oATLyIQRq19qjX1FxF7?usp=sharing)
- [Pytorch CIFAR10图像分类 LeNet5篇](https://blog.csdn.net/weixin_45508265/article/details/119305673)  [B站视频](https://www.bilibili.com/video/BV1FL411K7VJ)  [Colab Demo](https://colab.research.google.com/drive/15B0HBssfRzQk8mJyYF-v5fwAdPtNqf3H?usp=sharing) for LeNet5 [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/15B0HBssfRzQk8mJyYF-v5fwAdPtNqf3H?usp=sharing)
- [Pytorch CIFAR10图像分类 AlexNet篇](https://blog.csdn.net/weixin_45508265/article/details/119305848)  [B站视频](https://www.bilibili.com/video/BV1xu411B75x)  [Colab Demo](https://colab.research.google.com/drive/1d6CTYzyWeB03xiSlT8mzsZ_LtH9TlPvs?usp=sharing) for AlexNet [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1d6CTYzyWeB03xiSlT8mzsZ_LtH9TlPvs?usp=sharing)
- [Pytorch CIFAR10图像分类 VGG篇](https://blog.csdn.net/weixin_45508265/article/details/119332904)  [B站视频](https://www.bilibili.com/video/BV12L4y1u7WH)  [Colab Demo](https://colab.research.google.com/drive/1BO0wSY3w3xma-oATLyIQRq19qjX1FxF7?usp=sharing) for VGG16 [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1BO0wSY3w3xma-oATLyIQRq19qjX1FxF7?usp=sharing)
- [Pytorch CIFAR10图像分类 GoogLeNet篇](https://blog.csdn.net/weixin_45508265/article/details/119399239)  [B站视频](https://www.bilibili.com/video/BV1RS4y1274A)  [Colab Demo](https://colab.research.google.com/drive/1BO0wSY3w3xma-oATLyIQRq19qjX1FxF7?usp=sharing) for GoogLeNet Inceptionv1  [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1o8lfWHvr4WoyTA5Y9b4mSCSw2TEbXJb7?usp=sharing)
- [Pytorch CIFAR10图像分类 ResNet篇](https://blog.csdn.net/weixin_45508265/article/details/119532143) [B站视频](https://www.bilibili.com/video/BV1Wu411v72u)  [Colab Demo](https://colab.research.google.com/drive/1BO0wSY3w3xma-oATLyIQRq19qjX1FxF7?usp=sharing) for ResNet [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1W6d-eTY89bvGEL_QoMq4kw7m0dP9lHkS?usp=sharing)
- [Pytorch CIFAR10图像分类 DenseNet篇](https://blog.csdn.net/weixin_45508265/article/details/119648036)  [B站视频](https://www.bilibili.com/video/BV1ar4y1J77T)  [Colab Demo](https://colab.research.google.com/drive/1vkCEiDaOAb7TCfIErAekEikiN1Ll9IaS?usp=sharing) for DenseNet [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1vkCEiDaOAb7TCfIErAekEikiN1Ll9IaS?usp=sharing)
- [Pytorch CIFAR10图像分类 MobieNetv1篇](https://redamancy.blog.csdn.net/article/details/124636103) [Colab Demo](https://colab.research.google.com/drive/1r2umC8IoWfM5Qk0P8yCaUT6lyNKevMkK?usp=sharing) for MobileNetv1 [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1r2umC8IoWfM5Qk0P8yCaUT6lyNKevMkK?usp=sharing)
- [Pytorch CIFAR10图像分类 ResNeXt篇](https://redamancy.blog.csdn.net/article/details/126655797)  [Colab Demo](https://colab.research.google.com/drive/1BO0wSY3w3xma-oATLyIQRq19qjX1FxF7?usp=sharing) for ResNeXt [![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1vkCEiDaOAb7TCfIErAekEikiN1Ll9IaS?usp=sharing)
- [Pytorch CIFAR10 图像分类 Vision Transformer篇](https://redamancy.blog.csdn.net/article/details/126751948) for ViT  
- 

除此之外，所有的模型权重都在release之中，可以选择相对应的权重文件进行下载[模型权重](https://github.com/Dreaming-future/Pytorch-Image-Classification/releases/tag/v1.0.0)

- [Transer Learning](https://redamancy.blog.csdn.net/article/details/120213598)  [Colab Demo](https://colab.research.google.com/drive/1j7rg9eDbWnn8KJQvMVEaOgDiLUJqFxRh?usp=sharing)[![在这里插入图片描述](https://img-blog.csdnimg.cn/47fbca1712ba49719240c6dc3258ddc7.png)](https://colab.research.google.com/drive/1j7rg9eDbWnn8KJQvMVEaOgDiLUJqFxRh?usp=sharing)

  数据集也可以从[release](https://github.com/Dreaming-future/Pytorch-Image-Classification/releases/tag/v1.1.0)中获取

对于无法上github的同学，我们还可以通过Gitee来下载我们的代码和结果

https://github.com/Dreaming-future/Pytorch-Image-Classification/releases/tag/v1.0.0



# Comming soon 更新计划

- [x] MobileNetv1 Model 
- [x] MobileNetv2 Model
- [x] ResNeXt Model
- [x] Vision Transformer
- [ ] ShuffleNetv1 Model
- [ ] ShuffleNetv2 Model
- [ ] ZFNet Model
- [ ] SeNet Model
- [ ] Efficientent Model
- [ ] Swin Transformer



## 使用方法

下载`CIFAR10_code`里所有文件，直接运行ipynb即可，由于我是利用一个工具函数进行训练的，所以切记utils.py是必不可少的。

运行ipynb文件即可，对于网络的py文件会持续更新，之后会利用一个函数来选取对应的网络进行训练得到结果。



# <font face="Courier New" size=6>Data And Code

我的代码资源都在我的github和gitee上，大家有兴趣可以自提，CIFAR10可以利用代码下载，这里就不给出来了，当然也可以去官网。

- 路径1：[Gitee](https://gitee.com/DK-Jun/Pytorch-Image-Classification) [GitHub](https://github.com/Dreaming-future/Pytorch-Image-Classification)

- 路径2：百度网盘（网盘有时没有时常更新，多关注github）
  链接：[https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw ](https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw )
  提取码：5605 

  

除此之外，我还为图像分类这个专栏录了一下我的视频讲解，感兴趣的小伙伴可以来我的B站看视频进行学习，啃代码的时候，可能听一下也会有更多的感触哦
[https://space.bilibili.com/241286257/channel/seriesdetail?sid=2075039](https://space.bilibili.com/241286257/channel/seriesdetail?sid=2075039)

参考

- [Pytorch Note 快乐星球](https://blog.csdn.net/weixin_45508265/article/details/117809512)

