import numpy as np

import torch
import torch.nn.init as init
from torch.nn import Conv2d as Conv

try:
    from .prototype import NN
except:
    from prototype import NN


class Conv2d(NN):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=None, ratio=0.1):
        super(Conv2d, self).__init__()
        self.conv = Conv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.ratio = ratio
        self.weight_initialization()

    def weight_initialization(self):
        init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2))
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

    def create_shuffle_indices(self, x):
        _, in_planes, height, width = x.size()
        self.shuffle_until_here = int(in_planes * self.ratio)
        if self.shuffle_until_here >= 1:
            self.register_buffer('random_indices', torch.randperm(self.shuffle_until_here * height * width))

    def shuffle(self, x):
        if self.shuffle_until_here >= 1:
            shuffled_x, non_shuffled_x = x[:, :self.shuffle_until_here], x[:, self.shuffle_until_here:]
            batch, ch, height, width = shuffled_x.size()
            shuffled_x = torch.index_select(shuffled_x.view(batch, -1), 1, self.random_indices).view(batch, ch, height, width)
            return torch.cat((shuffled_x, non_shuffled_x), 1)
        else:
            return x

    def forward(self, x):
        if hasattr(self, 'random_indices') is False:
            self.create_shuffle_indices(x)
        x = self.shuffle(x)
        return self.conv(x)


if __name__ == '__main__':
    from torch.autograd import Variable
    net = Conv2d(12, 12, 3, 1, 1)
    y = net(Variable(torch.randn(2, 12, 32, 32)))
    print(y.size())
    net = Conv2d(3, 10, 3, 1, 1)
    y = net(Variable(torch.randn(2, 3, 32, 32)))
    print(y.size())
