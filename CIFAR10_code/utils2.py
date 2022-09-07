
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
    
    
    train_step = len(dataloader)
    with tqdm(total=train_step,desc=f'Train Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
        for step,(data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = get_acc(output,target)
            running_accuracy += acc
            running_loss += loss.data
            
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(**{'Train Acc' : running_accuracy.item()/(step+1),
                                'Train Loss' :running_loss.item()/(step+1),  
                                'Lr'   : lr})
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

                output = model(data)
                loss = criterion(output, target)
                acc = get_acc(output,target)
                
                test_accuracy += acc
                test_loss += loss.item()
                
                pbar.set_postfix(**{'Eval Acc' : test_accuracy.items()/(step+1),
                                'Eval Loss' :test_loss/(step+1)})
                pbar.update(1)
                
    test_loss, test_accuracy = test_loss/eval_step, test_accuracy/eval_step
    return test_loss, test_accuracy