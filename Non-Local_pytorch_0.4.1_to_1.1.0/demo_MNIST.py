import torch
import torch.utils.data as Data
import torchvision
from lib.network import Network
from torch import nn
import time


# def calc_acc(x, y):
#     x = torch.max(x, dim=-1)[1]
#     accuracy = sum(x == y) / x.size(0)
#     return accuracy


train_data = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root='./mnist/',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)

train_batch_num = len(train_loader)
test_batch_num = len(test_loader)

net = Network()
if torch.cuda.is_available():
    net = nn.DataParallel(net)
    net.cuda()

opt = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch_index in range(20):
    st = time.time()

    torch.set_grad_enabled(True)
    net.train()
    for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
            label_batch = label_batch.cuda()

        predict = net(img_batch)
        # acc = calc_acc(predict.cpu().data, label_batch.cpu().data)
        loss = loss_func(predict, label_batch)

        net.zero_grad()
        loss.backward()
        opt.step()

    print('(LR:%f) Time of a epoch:%.4fs' % (opt.param_groups[0]['lr'], time.time()-st))

    torch.set_grad_enabled(False)
    net.eval()
    total_loss = []
    total_acc = 0
    total_sample = 0

    for test_batch_index, (img_batch, label_batch) in enumerate(test_loader):
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
            label_batch = label_batch.cuda()

        predict = net(img_batch)
        loss = loss_func(predict, label_batch)

        predict = predict.argmax(dim=1)
        acc = (predict == label_batch).sum()

        total_loss.append(loss)
        total_acc += acc
        total_sample += img_batch.size(0)

    net.train()

    mean_acc = total_acc.item() * 1.0 / total_sample
    mean_loss = sum(total_loss) / total_loss.__len__()

    print('[Test] epoch[%d/%d] acc:%.4f%% loss:%.4f\n'
          % (epoch_index, 100, mean_acc * 100, mean_loss.item()))
