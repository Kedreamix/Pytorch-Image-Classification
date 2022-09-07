
'''
对训练函数进行更新
可视化更加方便，更加直观
'''
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1) # 得到概率
    correct = (pred_y == label).sum().data
    return correct / total

def plot_history(epochs, Acc = None, Loss=None, lr=None):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.style.use('seaborn')
    
    if Acc or Loss or lr:
        if not os.path.isdir('vis'):
            os.mkdir('vis')    
    epoch_list = range(1,epochs + 1)
    
    if Loss:
        plt.plot(epoch_list, Loss['train_loss'])
        plt.plot(epoch_list, Loss['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('Loss Value')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('vis/history_Loss.png')
        plt.show()

    if Acc:
        plt.plot(epoch_list, Acc['train_acc'])
        plt.plot(epoch_list, Acc['val_acc'])
        plt.xlabel('epoch')
        plt.ylabel('Acc Value')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('vis/history_Acc.png')
        plt.show()
    
    if lr:
        plt.plot(epoch_list, lr)
        plt.xlabel('epoch')
        plt.ylabel('Train LR')
        plt.savefig('vis/history_Lr.png')
        plt.show()
        
        
def train(epoch, epochs, model, dataloader, criterion, optimizer, scheduler = None):
    
    '''
    Function used to train the model over a single epoch and update it according to the
    calculated gradients.

    Args:
        model: Model supplied to the function
        dataloader: DataLoader supplied to the function
        criterion: Criterion used to calculate loss
        optimizer: Optimizer used update the model
        scheduler: Scheduler used to update the learing rate for faster convergence 
                   (Commented out due to poor results)
        resnet_features: Model to get Resnet Features for the hybrid architecture (Default=None)

    Output:
        running_loss: Training Loss (Float)
        running_accuracy: Training Accuracy (Float)
    '''
    running_loss = 0.0
    running_accuracy = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.train()
    train_step = len(dataloader)
    with tqdm(total=train_step,desc=f'Train Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
        for step,(data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            #---------------------
            #  释放内存
            #---------------------
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
        

            acc = get_acc(output,target)
            running_accuracy += acc.item()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
                
            pbar.set_postfix(**{'Train Acc' : running_accuracy/(step+1),
                                'Train Loss' :running_loss/(step+1)})
            pbar.update(1)
    if scheduler:
        scheduler.step(running_loss)
    running_loss, running_accuracy = running_loss/len(dataloader), running_accuracy/len(dataloader)
    return running_loss, running_accuracy


def evaluation(epoch, epochs, model, dataloader, criterion):
    '''
    Function used to evaluate the model on the test dataset.

    Args:
        model: Model supplied to the function
        dataloader: DataLoader supplied to the function
        criterion: Criterion used to calculate loss
        resnet_features: Model to get Resnet Features for the hybrid architecture (Default=None)
    
    Output:
        test_loss: Testing Loss (Float)
        test_accuracy: Testing Accuracy (Float)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_step = len(dataloader)
    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        with tqdm(total=eval_step,desc=f'Evaluation Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,(data, target) in enumerate(dataloader):
                data = data.to(device)
                target = target.to(device)
                #---------------------
                #  释放内存
                #---------------------
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                output = model(data)
                
                loss = criterion(output, target)
                acc = get_acc(output,target)
                
                test_accuracy += acc.item()
                test_loss += loss.item()
                
                pbar.set_postfix(**{'Eval Acc' : test_accuracy/(step+1),
                                'Eval Loss' :test_loss/(step+1)})
                pbar.update(1)
                
    test_loss, test_accuracy = test_loss/eval_step, test_accuracy/eval_step
    return test_loss, test_accuracy

def test(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct = 0   # 定义预测正确的图片数，初始化为0
    total = 0     # 总共参与测试的图片数，也初始化为0
    model.eval()
    with torch.no_grad():
        for data in dataloader:  # 循环每一个batch
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            model.eval()  # 把模型转为test模式
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = model(images)  # 输入网络进行测试

            # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)          # 更新测试图片的数量
            correct += (predicted == labels).sum() # 更新正确分类的图片的数量

    print('Accuracy of the network on the %d test images: %.2f %%' % (total, 100 * correct / total))


def test_precls(model, dataloader, classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 定义2个存储每类中测试正确的个数的 列表，初始化为0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True, num_workers=2)
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            c = (predicted == labels).squeeze()
            for i in range(len(images)):      # 因为每个batch都有4张图片，所以还需要一个4的小循环
                label = labels[i]   # 对各个类的进行各自累加
                class_correct[label] += c[i]
                class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))