from scipy.stats import normaltest
import numpy as np
import torch as t
import random, time

import torch.nn.functional as F
import torch.optim as optim
from torch import nn

# 定义一些超参数
#learning_rate = 0.01
num_epoches = 20
lr = 0.0001
momentum = 0.1

dtype = t.float
with open('rec','r+') as f:
    lines = f.readlines()
train_loader_x = []
train_loader_y = []
train_loader_z = []
for line in lines:
    l = line.strip('\n').split(' ')
    for i in range(6):
        l[i] = [float(l[i])]
    train_loader_x.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[3]], dtype=dtype)))
    train_loader_y.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[4]], dtype=dtype)))
    train_loader_z.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[5]], dtype=dtype)))
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

class Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Net, self).__init__()
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

#检测是否有可用的GPU，有则使用，否则使用CPU
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#实例化网络
model_x = Net(3, 500, 1000, 100, 1)
model_y = Net(3, 500, 1000, 100, 1)
model_z = Net(3, 500, 1000, 100, 1)
for i in range(20, 0, -1):
    try:
        model_x.load_state_dict(t.load(f'model_x_{i-1}.mdl'))
        print(f'---------- Model_x num. {i-1} loaded. ---------')
        model_y.load_state_dict(t.load(f'model_y_{i-1}.mdl'))
        print(f'---------- Model_y num. {i-1} loaded. ---------')
        model_z.load_state_dict(t.load(f'model_z_{i-1}.mdl'))
        print(f'---------- Model_z num. {i-1} loaded. ---------')
        break
    except:
        pass
model_x.to(device)
model_y.to(device)
model_z.to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer_x = optim.SGD(model_x.parameters(), lr=lr, momentum=momentum)
optimizer_y = optim.SGD(model_y.parameters(), lr=lr, momentum=momentum)
optimizer_z = optim.SGD(model_z.parameters(), lr=lr, momentum=momentum)

# 开始训练
for epoch in range(num_epoches):
    train_loss_x = 0
    train_loss_y = 0
    train_loss_z = 0
    model_x.train()
    model_y.train()
    model_z.train()
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
        '''
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == facc).sum().item()
        acc = num_correct / eul.shape[0]
        train_acc_x += acc
        '''
    #losses_x.append(train_loss_x / len(train_loader_x))
    for eul, facc in train_loader_y:
        eul=eul.to(device)
        facc = facc.to(device)
        #eul = eul.view(eul.size(0), -1)
        # 前向传播
        out = model_y(eul)
        loss = criterion(out, facc)
        # 反向传播
        optimizer_y.zero_grad()
        loss.backward()
        optimizer_y.step()
        # 记录误差
        train_loss_y += loss.item()
    #losses_y.append(train_loss_y / len(train_loader_y))
    for eul, facc in train_loader_z:
        eul=eul.to(device)
        facc = facc.to(device)
        #eul = eul.view(eul.size(0), -1)
        # 前向传播
        out = model_z(eul)
        loss = criterion(out, facc)
        # 反向传播
        optimizer_z.zero_grad()
        loss.backward()
        optimizer_z.step()
        # 记录误差
        train_loss_z += loss.item()
    #losses_z.append(train_loss_z / len(train_loader_z))

    # 在测试集上检验效果
    eval_loss_x = 0
    eval_loss_y = 0
    eval_loss_z = 0
    # 将模型改为预测模式
    model_x.eval()
    model_y.eval()
    model_z.eval()
    for eul, facc in test_loader_x:
        eul=eul.to(device)
        facc = facc.to(device)
        out = model_x(eul)
        loss = criterion(out, facc)
        # 记录误差
        eval_loss_x += loss.item()
    #eval_losses_x.append(eval_loss_x / len(test_loader_x))
    for eul, facc in test_loader_y:
        eul=eul.to(device)
        facc = facc.to(device)
        out = model_y(eul)
        loss = criterion(out, facc)
        # 记录误差
        eval_loss_y += loss.item()
    #eval_losses_y.append(eval_loss_y / len(test_loader_y))
    for eul, facc in test_loader_z:
        eul=eul.to(device)
        facc = facc.to(device)
        out = model_z(eul)
        loss = criterion(out, facc)
        # 记录误差
        eval_loss_z += loss.item()
    #eval_losses_z.append(eval_loss_z / len(test_loader_x))

    print('epoch: {}, Train Loss x: {:.4f}, Train Loss y: {:.4f}, Train Loss z: {:.4f}, Test Loss x: {:.4f}, Test Loss y: {:.4f}, Test Loss z: {:.4f}'
        .format(epoch, train_loss_x / len(train_loader_x), train_loss_y / len(train_loader_y), train_loss_z / len(train_loader_z), 
            eval_loss_x / len(test_loader_x), eval_loss_y / len(test_loader_y), eval_loss_z / len(test_loader_z)))
    random.seed(time.time())
    x, y, z = random.choice(train_loader_x), random.choice(train_loader_y), random.choice(train_loader_z)
    print(f'Example: {x[1]} {y[1]} {z[1]} Out: {model_x(x[0].to(device))} {model_y(y[0].to(device))} {model_z(z[0].to(device))}')

    t.save(model_x.state_dict(), f'model_x_{epoch}.mdl')
    t.save(model_y.state_dict(), f'model_y_{epoch}.mdl')
    t.save(model_z.state_dict(), f'model_z_{epoch}.mdl')
