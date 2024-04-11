from scipy.stats import normaltest
import numpy as np
import torch as t
import random, time
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from basic_functions import *

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1) #,nn.BatchNorm1d(n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2) #,nn.BatchNorm1d (n_hidden_2))
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3) #,nn.BatchNorm1d (n_hidden_3))
        self.layer4 = nn.Linear(n_hidden_3, n_hidden_4) #,nn.BatchNorm1d (n_hidden_4))
        self.layer5 = nn.Linear(n_hidden_4, out_dim)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.sigmoid(self.layer5(x))
        return x.squeeze()
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = t.tensor(x).float().to(device)
        self.y = t.tensor(y).float().to(device)
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    def __len__(self): 
        return len(self.x)


def training_set(seg_length, offset, training_set_length, mode) -> tuple:
    trainSet, testSet = load_train_data(seg_length, offset, training_set_length)
    print(random.choice(trainSet), '\n', random.choice(trainSet), '\n', random.choice(testSet), '\n', random.choice(testSet))
    for i in range(len(trainSet)):
        if trainSet[i][1] != [mode]:
            trainSet[i][1] = 0
        else:
            trainSet[i][1] = 1
    for i in range(len(testSet)):
        if testSet[i][1] != [mode]:
            testSet[i][1] = 0
        else:
            testSet[i][1] = 1

    train_ds = MyDataset([x[0] for x in trainSet], [x[1] for x in trainSet])
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    test_ds = MyDataset([x[0] for x in testSet], [x[1] for x in testSet])
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

    return train_dl, test_dl

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
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'rms':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        print(f'Wrong optimizer type: {optimizer_type}')
        exit(1)
    return criterion, optimizer

def training(model, criterion, optimizer, train_loader, test_loader, num_epoches, pt_path, threshold) -> None:
    # 开始训练
    prev_train_loss = []
    prev_test_loss = []
    for epoch in range(num_epoches):
        print(f'On number {epoch} training')
        train_loss = 0
        model.train()

        #动态修改参数学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= lr_itr_rate

        for seg, tag in train_loader:
            # 前向传播
            out = model(seg)
            loss = criterion(out, tag)
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
            out = model(seg)
            loss = criterion(out, tag)
            test_loss += loss.item()

        print('epoch: {}, Train Loss x: {:.4f}, Test Loss x: {:.4f}'
            .format(epoch, train_loss / len(train_loader), test_loss / len(test_loader)))
        
        t.save(model.state_dict(), pt_path)
        try:
            if abs(train_loss - prev_train_loss[-1]) / prev_train_loss[-1] < threshold and abs(test_loss - prev_test_loss[-1]) / prev_test_loss[-1] < threshold:
                print('Converged.')
                break
            elif train_loss == 0 and test_loss == 0:
                break
        except:
            pass
        prev_train_loss.append(train_loss)
        prev_test_loss.append(test_loss)
    return

def testing(data, model):
    for seg, tag in data:
        out = model(seg)
        print(f'model out: {out}, actual val: {tag}')
    return

if __name__ == "__main__":

    num_epoches = 20
    learning_rate = 0.01
    momentum = 0.1
    lr_itr_rate = 1
    threshold = 0.01

    seg_length = 3
    offset = 5
    training_set_length = 40
    mode = 1
    criterion_type = 'mse'
    optimizer_type = 'adagrad'
    
    pt_path = f'net\\model_nn_{learning_rate}_{momentum}_{criterion_type}_{optimizer_type}.pt'
    dtype = t.float
    train_loader, test_loader = training_set(seg_length, offset, training_set_length, mode)
    
    model = NeuralNet(1, 300, 800, 400, 1, 1)
    try:
        model.load_state_dict(t.load(pt_path))
        print(f'model parameters loaded.')
    except Exception as e:
        print(f'Exception: {e}\nStart with new model.')

    criterion, optimizer = cri_opt(criterion_type, optimizer_type, model, learning_rate, momentum)

    model.to(device)
    print(f'Device: {device}')

    training(model, criterion, optimizer, train_loader, test_loader, num_epoches, pt_path, threshold)
    testing(test_loader, model)