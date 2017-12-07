import logging
import torch
import torch.utils.data as Data
import torchvision
from lib.network import Network
from torch.autograd import Variable
from torch import nn
import config as cfg
from lib.utils import create_architecture


def calc_acc(x, y):
    x = torch.max(x, dim=-1)[1]
    accuracy = sum(x == y) / x.size(0)
    return accuracy

logging.getLogger().setLevel(logging.INFO)

create_architecture()

train_data = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root='./mnist/',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False)

train_batch_num = len(train_loader)
test_batch_num = len(test_loader)

net = Network()
if torch.cuda.is_available():
    net.cuda()

opt = torch.optim.Adam(net.parameters(), lr=cfg.LR, weight_decay=cfg.weight_decay)
loss_func = nn.CrossEntropyLoss()

if cfg.load_model:
    net.load_state_dict(torch.load(cfg.model_path))

for epoch_index in range(cfg.epoch):
    for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
        img_batch = Variable(img_batch)
        label_batch = Variable(label_batch)

        if torch.cuda.is_available():
            img_batch = img_batch.cuda(cfg.cuda_num)
            label_batch = label_batch.cuda(cfg.cuda_num)

        predict = net(img_batch)
        acc = calc_acc(predict.cpu().data, label_batch.cpu().data)
        loss = loss_func(predict, label_batch)

        net.zero_grad()
        loss.backward()
        opt.step()

        # logging.info('epoch[%d/%d] batch[%d/%d] loss:%.4f acc:%.4f' %
        #              (epoch_index, cfg.epoch, train_batch_index, train_batch_num, loss.data[0], acc))

    opt.param_groups[0]['lr'] = cfg.LR * (cfg.LR_decay_rate ** (epoch_index // cfg.LR_decay_every_epoch))
    print('LR', opt.param_groups[0]['lr'])
# if (train_batch_index + 1) % cfg.test_per_batch == 0:

    net.eval()

    total_loss = 0
    total_acc = 0

    for test_batch_index, (img_batch, label_batch) in enumerate(test_loader):
        img_batch = Variable(img_batch, volatile=True)
        label_batch = Variable(label_batch, volatile=True)

        if torch.cuda.is_available():
            img_batch = img_batch.cuda(cfg.cuda_num)
            label_batch = label_batch.cuda(cfg.cuda_num)

        predict = net(img_batch)
        acc = calc_acc(predict.cpu().data, label_batch.cpu().data)
        loss = loss_func(predict, label_batch)

        total_loss += loss
        total_acc += acc

    net.train()

    mean_acc = total_acc / test_batch_num
    mean_loss = total_loss / test_batch_num
    logging.info('[Test] epoch[%d/%d] acc:%.4f loss:%.4f '
                 % (epoch_index, cfg.epoch, mean_acc, mean_loss.data[0]))

    torch.save(net.state_dict(), cfg.model_path)
