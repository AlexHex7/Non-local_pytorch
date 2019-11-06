import torch
import torch.utils.data as Data
import torchvision
from lib.network import Network
from torch import nn
import numpy as np


test_data = torchvision.datasets.MNIST(root='./mnist/',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)

test_loader = iter(Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False))

net = Network()
if torch.cuda.is_available():
    net = nn.DataParallel(net)
    net.cuda()

net.load_state_dict(torch.load('weights/net.pth'))


img_batch, label_batch = test_loader.__next__()
img_batch = img_batch.cuda()
label_batch = label_batch.cuda()

torch.set_grad_enabled(False)
net.eval()

_, nl_mep_list = net.module.forward_with_nl_map(img_batch)

# (b, h1*w1, h2*w2)
nl_map_1 = nl_mep_list[0].cpu().numpy()
nl_map_2 = nl_mep_list[1].cpu().numpy()

img = torchvision.transforms.ToPILImage()(img_batch.cpu()[0])
img.save('nl_map_vis/sample.png')
np.save('nl_map_vis/nl_map_1', nl_map_1)
np.save('nl_map_vis/nl_map_2', nl_map_2)
