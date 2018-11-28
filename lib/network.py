from torch import nn
# from lib.non_local_concatenation import NONLocalBlock2D
# from lib.non_local_gaussian import NONLocalBlock2D
from lib.non_local_embedded_gaussian import NONLocalBlock2D
# from lib.non_local_dot_product import NONLocalBlock2D


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.convs(x).view(batch_size, -1)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    import torch

    img = torch.randn(3, 1, 28, 28)
    net = Network()
    out = net(img)
    print(out.size())

