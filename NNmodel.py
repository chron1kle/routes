from scipy.stats import normaltest
import numpy as np
import torch as t
import random, time
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from basic_functions import *
from Kmeans import KMeans

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1) #,nn.BatchNorm1d(n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2) #,nn.BatchNorm1d (n_hidden_2))
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, n_hidden_3)
        self.layer5 = nn.Linear(n_hidden_3, n_hidden_3) #,nn.BatchNorm1d (n_hidden_3))
        self.layer6 = nn.Linear(n_hidden_3, n_hidden_4) #,nn.BatchNorm1d (n_hidden_4))
        self.layer7 = nn.Linear(n_hidden_4, out_dim)
    def forward(self, x):
        x = F.relu(self.layer1(x.view(-1)))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.sigmoid(self.layer7(x))
        return x
    
class ConvNet(nn.Module):
    def __init__(self, para):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential( #input shape (3, 4)
            nn.Conv2d(in_channels=1, #input height 必须手动提供 输入张量的channels数
                      out_channels=para, #n_filter 必须手动提供 输出张量的channels数
                      kernel_size=3, #filter size 必须手动提供 卷积核的大小 
                      # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
                      stride=1, #filter step 卷积核在图像窗口上每次平移的间隔，即所谓的步长
                      padding=2 #con2d出来的图片大小不变 Pytorch与Tensorflow在卷积层实现上最大的差别就在于padding上
            ), # output shape (16,28,28) 输出图像尺寸计算公式是唯一的 # O = （I - K + 2P）/ S +1
            nn.ReLU(), # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool2d(kernel_size=2) #2x2采样，28/2=14，output shape (6,2) maxpooling有局部不变性而且可以提取显著特征的同时降低模型的参数，从而降低模型的过拟合
        ) 
        self.conv2 = nn.Sequential(nn.Conv2d(para, 3, 2, 1, 1), #output shape (3, 2)
                                  nn.ReLU(),
                                  nn.MaxPool2d(2))
        self.out = nn.Linear(6, 1)

    def forward(self, x):
        x = self.conv1(x) # 卷一次
        x = self.conv2(x) # 卷两次
        x = x.view(x.size(0), -1)
        return F.sigmoid(self.out(x.view(-1)))

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = t.tensor(x).float().to(device)
        self.y = t.tensor(y).float().to(device)
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    def __len__(self): 
        return len(self.x)


def training_set(trainSet, testSet, mode) -> tuple:  # seg_length, offset, training_set_length
    # trainSet, testSet = load_train_data(seg_length, offset, training_set_length, mode)

    
    for i in range(len(trainSet)):
        if trainSet[i][1] != mode:
            trainSet[i][1] = 0
        else:
            trainSet[i][1] = 1
    for i in range(len(testSet)):
        if testSet[i][1] != mode:
            testSet[i][1] = 0
        else:
            testSet[i][1] = 1

    train_ds = MyDataset([x[0] for x in trainSet], [x[1] for x in trainSet])
    train_dl = DataLoader(train_ds, batch_size=1)

    test_ds = MyDataset([x[0] for x in testSet], [x[1] for x in testSet])
    test_dl = DataLoader(test_ds, batch_size=1)

    # for seg, tag in testSet:
    #     if tag == 1:
    #         print(seg)
    # for seg, tag in test_dl:
    #     if tag == t.tensor([1]):
    #         print(seg)
    # print(i)

    return train_dl, test_dl

def labelling_set(seg_length, offset, flag) -> object:
    dataSet = load_seg_data(seg_length, offset, flag=flag)
    label_ds = MyDataset([x[1] for x in dataSet], [x[2] for x in dataSet])
    label_dl = DataLoader(label_ds, batch_size=1)
    return label_dl

def cri_opt(criterion_type, optimizer_type, model, learning_rate, momentum) -> object:
    if criterion_type == 'mse':
        criterion = nn.MSELoss()
    elif criterion_type == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        print(f'Wrong criterion type: {criterion_type}')
        exit(1)
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rms':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print(f'Wrong optimizer type: {optimizer_type}')
        exit(1)
    return criterion, optimizer

def training(model, criterion, optimizer, train_loader, test_loader, num_epoches, pt_path, threshold) -> None:
    # 开始训练
    prev_train_loss = []
    prev_test_loss = []
    for epoch in range(num_epoches):
        print(f'\nOn number {epoch} training')
        train_loss = 0
        model.train()

        #动态修改参数学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= lr_itr_rate

        for seg, tag in train_loader:
            # 前向传播
            try:
                out = model(seg)
                loss = criterion(out, tag)
            except ValueError as e:
                print(f'\033[31mtrain set ValueError: {e}\nout: {out}\ntag: {tag}\033[30m')
                exit(1)
            except RuntimeError as e:
                print(f'\033[31mtrain set RuntimeError: {e}\nseg: {seg}\ntag: {tag}\033[30m')
                exit(1)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 在测试集上检验效果
        test_loss = 0

        # 将模型改为预测模式
        model.eval()

        for seg, tag in test_loader:
            try:
                out = model(seg)
            except RuntimeError as e:
                print(f'test set Error: {e}\nseg: {seg}\ntag: {tag}')
                # exit(1)
            # out = model(seg)
            loss = criterion(out, tag)
            # print(out, '\n', tag, '\n', loss)
            test_loss += loss.item()

        unit_train_loss = train_loss / len(train_loader)
        unit_test_loss = test_loss / len(test_loader)
        print('epoch: {}, Train Loss x: {:.4f}, Test Loss x: {:.4f}'
            .format(epoch, train_loss, test_loss))
        
        t.save(model, pt_path)
        try:
            if round(unit_train_loss, 4) == 0 and round(unit_test_loss, 4) == 0:
                pass # return
            elif abs(train_loss - prev_train_loss[-1]) / prev_train_loss[-1] < threshold and abs(test_loss - prev_test_loss[-1]) / prev_test_loss[-1] < threshold:
                print('Converged.')
                return
        except:
            pass
        prev_train_loss.append(train_loss)
        prev_test_loss.append(test_loss)
    return

def testing(data, model):
    sum = 0
    l = []
    for seg, tag in data:
        out = model(seg).item()
        sum += (out - tag.item()) ** 2
        l.append((round(out, 6), int(tag.item())))
    l.sort(key=lambda x: x[0])
    for out, tag in l:
        if tag == 1:
            print('\033[31m')
            log_write(f'model out: {out}, actual val: {tag}')
            print('\033[30m')
        else:
            log_write(f'model out: {out}, actual val: {tag}')
    log_write(f'Total loss: {sum / len(data)}')
    return

def confidence_labelling(data, model) -> object:
    for i in range(len(data)):
        data[i].append(round(model(data[i][1]).item(), 6))
    return data

if __name__ == "__main__":

    while True:
        userIn = int(input("Input flag: "))
        if userIn not in [0, 1, 2]:
            continue
        flag = ["training", "testing", "labelling"][userIn]
        if userIn == 2:
            model_folder = 'trainedNet'
        else:
            model_folder = 'net'
        break

    num_epoches = 2
    learning_rate = 0.1
    momentum = 0.1
    lr_itr_rate = 1
    threshold = 0.005
    cnn_para = 1000

    seg_length = 3
    offset = 5
    testing_set_length = 30

    mode = 3
    model_type = 'nn'
    criterion_type = 'ce'
    optimizer_type = 'adagrad'
    
    log_path = f'ModelLog.log'
    dtype = t.float
    
    log_write('-----------------------------------------', log_path)
    log_write(f'Model\ntype: {model_type}\ncriterion: {criterion_type}\noptimizer: {optimizer_type}\nlearning_rate: {learning_rate}\nthreshold: {threshold}\nsegmentation length: {seg_length}\noffset: {offset}\ntesting set length: {testing_set_length}', log_path)

    try:
        
        if model_type == 'nn':
            # model = NeuralNet(seg_length * 4, 600, 1400, 800, 300, 1)
            pt_path = f'{model_folder}\\{mode}_model_{model_type}_{criterion_type}_{optimizer_type}_{learning_rate}_.mdl'
        elif model_type == 'cnn':
            # model = ConvNet(cnn_para)
            pt_path = f'{model_folder}\\{mode}_model_{model_type}_{criterion_type}_{optimizer_type}_{learning_rate}_{cnn_para}_.mdl'
        else:
            print(f'Wrong model type: {model_type}')
            exit(0)
        if flag == "labelling" or flag == "testing":
            model = t.load(pt_path)
            print(f'model loaded.')
    except Exception as e:
        if flag == "labelling":
            print(f'Wrong parameters.')
            exit(1)
        

    if flag == "training":
        if model_type == 'nn':
            model = NeuralNet(seg_length, 600, 1400, 800, 300, 1)
            # pt_path = f'{model_folder}\\{mode}_model_{model_type}_{criterion_type}_{optimizer_type}_{learning_rate}_.mdl'
        elif model_type == 'cnn':
            model = ConvNet(cnn_para)
            # pt_path = f'{model_folder}\\{mode}_model_{model_type}_{criterion_type}_{optimizer_type}_{learning_rate}_{cnn_para}_.mdl'
        print(f'Start with new model.')

    model.to(device)
    print(f'Device: {device}')

    criterion, optimizer = cri_opt(criterion_type, optimizer_type, model, learning_rate, momentum)

    if flag == "training":
        dset = LoadCatagory('312', dates)
        dset.segment(seg_length)
        train_set, test_set = dset.layout(testing_set_length) # training_set(seg_length, offset, testing_set_length, mode)
        train_loader, test_loader = training_set(train_set, test_set, mode)
        training(model, criterion, optimizer, train_loader, test_loader, num_epoches, pt_path, threshold)
        testing(test_loader, model)
    elif flag == "testing":
        train_loader, test_loader = training_set(seg_length, offset, testing_set_length, mode)
        testing(test_loader, model)
    elif flag == "labelling":
        data_loader = labelling_set(seg_length, offset, flag)
        save_confi_data(confidence_labelling(data_loader, model), seg_length, offset)
    else:
        print(f'Wrong flag: {flag}')
        exit(1)

    print(f'file path: {pt_path}')