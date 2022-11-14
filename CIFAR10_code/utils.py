import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm # 在脚本中使用较好
# from tqdm import tqdm_notebook as tqdm # 在jupyter中使用


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1) # 得到概率
    correct = (pred_y == label).sum().data
    return torch.div(correct, total)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def train(net, trainloader, testloader, epochs, optimizer , criterion, scheduler , path = './model.pth', writer = None ,verbose = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    train_acc_list, test_acc_list = [],[]
    train_loss_list, test_loss_list = [],[]
    lr_list  = []
    for i in range(epochs):
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        if torch.cuda.is_available():
            net = net.to(device)
        net.train()
        train_step = len(trainloader)
        with tqdm(total=train_step,desc=f'Train Epoch {i + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,data in enumerate(trainloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                # 释放内存
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # formard
                outputs = net(im)
                loss = criterion(outputs,label)
                # backward
                loss.backward()
                # 更新参数
                optimizer.step()
                # 累计损失
                train_loss += loss.item()
                train_acc += get_acc(outputs,label).item()
                pbar.set_postfix(**{'Train Acc' : train_acc/(step+1),
                                'Train Loss' :train_loss/(step+1)})
                pbar.update(1)
            pbar.close()
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        # 更新学习率
        scheduler.step(train_loss)
        if testloader is not None:
            net.eval()
            test_step = len(testloader)
            with torch.no_grad():
                with tqdm(total=test_step,desc=f'Test Epoch {i + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
                    for step,data in enumerate(testloader,start=0):
                        im,label = data
                        im = im.to(device)
                        label = label.to(device)
                        # 释放内存
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        outputs = net(im)
                        loss = criterion(outputs,label)
                        test_loss += loss.item()
                        test_acc += get_acc(outputs,label).item()
                        pbar.set_postfix(**{'Test Acc' : test_acc/(step+1),
                                'Test Loss' :test_loss/(step+1)})
                        pbar.update(1)
                    pbar.close()
                test_loss = test_loss / len(testloader)
                test_acc = test_acc * 100 / len(testloader)
            if verbose:
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            print(
                'Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epochs, train_loss, train_acc, test_loss, test_acc,lr))
        else:
            print('Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(i+1,epochs,train_loss,train_acc,lr))

        # ====================== 使用 tensorboard ==================
        if writer is not None:
            writer.add_scalars('Loss', {'train': train_loss,
                                    'test': test_loss}, i+1)
            writer.add_scalars('Acc', {'train': train_acc ,
                                   'test': test_acc}, i+1)
            writer.add_scalar('Learning Rate',lr,i+1)
        # =========================================================
        # 如果取得更好的准确率，就保存模型
        if test_acc > best_acc:
            torch.save(net,path)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Lr = lr_list
    return Acc, Loss, Lr

def plot_history(epochs, Acc, Loss, lr):
    plt.rcParams['figure.figsize'] = (12.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1,epochs + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['test_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['test_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(epoch_list, lr)
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.show()

