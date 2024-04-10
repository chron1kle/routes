from scipy.stats import normaltest
import numpy as np
import torch as t
import random, time

import torch.nn.functional as F
import torch.optim as optim
from torch import nn



class NeuralNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),nn.BatchNorm1d (n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),nn.BatchNorm1d (n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
    
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


if __name__ == "__main__":

    # 定义一些超参数
    #learning_rate = 0.01
    num_epoches = 20
    lr = 0.0001
    momentum = 0.1

    dtype = t.float

    train_loader_x = []

    for line in lines:
        l = line.strip('\n').split(' ')
        for i in range(6):
            l[i] = [float(l[i])]
        train_loader_x.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[3]], dtype=dtype)))
    with open('test','r+') as f:
        lines = f.readlines()
    test_loader_x = []
    test_loader_y = []
    test_loader_z = []
    for line in lines:
        l = line.strip('\n').split(' ')
        for i in range(6):
            l[i] = [float(l[i])]
        test_loader_x.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[3]], dtype=dtype)))
        test_loader_y.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[4]], dtype=dtype)))
        test_loader_z.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[5]], dtype=dtype)))
    #检测是否有可用的GPU，有则使用，否则使用CPU
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    #实例化网络
    model_x = Net(3, 500, 1000, 100, 1)
    '''
    for i in range(20, 0, -1):
        try:
            model_x.load_state_dict(t.load(f'model_x_{i-1}.mdl'))
            print(f'---------- Model_x num. {i-1} loaded. ---------')

            break
        except:
            pass
    '''

    model_x.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_x = optim.SGD(model_x.parameters(), lr=lr, momentum=momentum)


    # 开始训练
    for epoch in range(num_epoches):
        train_loss_x = 0

        model_x.train()

        #动态修改参数学习率
        if epoch%5==0:
            optimizer_x.param_groups[0]['lr']*=0.1

        for eul, facc in train_loader_x:
            eul=eul.to(device)
            facc = facc.to(device)
            #eul = eul.view(eul.size(0), -1)
            # 前向传播
            out = model_x(eul)
            loss = criterion(out, facc)
            # 反向传播
            optimizer_x.zero_grad()
            loss.backward()
            optimizer_x.step()
            # 记录误差
            train_loss_x += loss.item()

        # 在测试集上检验效果
        eval_loss_x = 0

        # 将模型改为预测模式
        model_x.eval()

        for eul, facc in test_loader_x:
            eul=eul.to(device)
            facc = facc.to(device)
            out = model_x(eul)
            loss = criterion(out, facc)
            # 记录误差
            eval_loss_x += loss.item()


        print('epoch: {}, Train Loss x: {:.4f}, Test Loss x: {:.4f}'
            .format(epoch, train_loss_x / len(train_loader_x), eval_loss_x / len(test_loader_x)))
        random.seed(time.time())
        x = random.choice(train_loader_x)

        t.save(model_x.state_dict(), f'model_x_{epoch}.mdl')

