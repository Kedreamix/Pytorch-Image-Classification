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
            	--optimizer Adam # 优化器 如Adam,AdamW,SGD
                --batch-size 64 # batch size
                --verbose # 可视化
                --logs # 日志，每次都保存模型
```

