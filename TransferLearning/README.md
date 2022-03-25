# Train your data in Pytorch

## Taining

实际上使用方法是很简单的，如果下载了数据集之后，就可以直接python train.py

```python
start training with:
python train.py

You can manually resume the training with:
python train.py --data  './cats_and_dogs_filtered/'
		--model 'ResNet' # choices ['AlexNet','VGG','ResNet','MobileNet','ShuffleNet','DenseNet','MnasNet']
		--classes 2 # set classes 
    		--epochs 10 # set epochs
        	--lr  0.02 # set learning rate
            	--optimizer 'Adam' # 优化器 如Adam,AdamW,SGD
                --batch-size 64 # batch size
                --verbose # 可视化
                --logs # 日志，每次都保存模型
```

## 使用自己的数据集

这里要注意，对于自己的数据集，需要按照一定的排列格式，可以按照你的数据集排列方式，最后在运行的时候改成python train.py --data './data'即可

```txt
data
	----train
    		----dogs
         	----cats
         	----...
    ----validation
    		----dogs
         	----cats
         	----...
   	----test(放任意图片，可以在命令行设置，没有默认0)
```

