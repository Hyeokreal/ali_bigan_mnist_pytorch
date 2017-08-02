import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# for test
import torch
from torch.autograd import Variable

z_dim = 100
x_dim = 32

batch = 100

std = 0.01

'''
init

nninit.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
nninit.constant(self.conv1.bias, 0.1)
'''


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = list()
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = list()
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def linear(l_in, l_out, bn=True):
    layers = list()
    layers.append(nn.Linear(l_in, l_out))
    if bn:
        layers.append(nn.BatchNorm1d(l_out))
    return nn.Sequential(*layers)


# gx
class P(nn.Module):
    def __init__(self, d=16):

        super(P, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)

        self.weight_init(mean=0.0, std=0.02)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         m.weight.data.normal_(0.0, std)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, std)
        #         m.bias.data.zero_()


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        # print("p shape in ", z.size())
        out = F.leaky_relu(self.deconv1_bn(self.deconv1(z)))
        # print("p shape ", out.size())
        out = F.leaky_relu(self.deconv2_bn(self.deconv2(out)))
        # print("p shape ", out.size())
        out = F.leaky_relu(self.deconv3_bn(self.deconv3(out)))
        # print("p shape here ", out.size())
        out = F.tanh(self.deconv4(out))
        # print("p shape out", out.size())
        return out


# gz
class Q(nn.Module):
    def __init__(self, d=16):
        super(Q, self).__init__()
        self.conv1 = nn.Conv2d(1, d * 2, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d * 2)
        self.conv2 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 4)
        self.conv3 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 8)
        self.conv4 = nn.Conv2d(d * 8, z_dim * 2, 5, 2, 1)
        # self.conv4_bn = nn.BatchNorm2d(z_dim * 8)
        # self.conv5 = nn.Conv2d()

        # self.mu = nn.Conv2d(d * 8, z_dim, 1, 1, 0)
        # self.sigma = nn.Conv2d(d * 8, z_dim, 1, 1, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         m.weight.data.normal_(0.0, std)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, std)
        #         m.bias.data.zero_()

        self.weight_init(mean=0.0, std=0.02)

    def get_e(self):
        return torch.randn([batch, z_dim, 1, 1])

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        # print("q x in size : ", x.size())
        out = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        # print("q shape: ", out.size())
        out = F.leaky_relu(self.conv2_bn(self.conv2(out)))
        # print("q shape: ", out.size())
        out = F.leaky_relu(self.conv3_bn(self.conv3(out)))
        # print("q shape: ", out.size())
        out = self.conv4(out)
        # print("q shape: ", out.size())
        # mu = self.mu(out)
        # print("mu: ", mu.size())
        # sig = self.sigma(out)
        # print("sig: ", sig.size())
        # e = Variable(self.get_e())
        # e.requires_grad = False
        # log_sig = torch.exp(sig) * e

        # out = mu + log_sig

        # out = F.sigmoid(self.conv4_bn(self.conv4(out)))
        # print("q shape out", out.size())
        return out


class D(nn.Module):
    # initializers
    def __init__(self, d=16):
        super(D, self).__init__()

        # dx part
        self.dx_conv1 = nn.Conv2d(1, 32, 5, 1, 0)
        self.dx_conv1_bn = nn.BatchNorm2d(32)
        self.dx_conv2 = nn.Conv2d(32, 64, 4, 2, 0)
        self.dx_conv2_bn = nn.BatchNorm2d(64)
        self.dx_conv3 = nn.Conv2d(64, 128, 4, 2, 0)
        self.dx_conv3_bn = nn.BatchNorm2d(128)
        self.dx_conv4 = nn.Conv2d(128, 256, 4, 2, 0)
        self.dx_conv4_bn = nn.BatchNorm2d(256)
        # dz part

        self.dz_conv1 = nn.Conv2d(z_dim, 256, 1, 1, 0)
        self.dz_conv2 = nn.Conv2d(256, 256, 1, 1, 0)

        # dxz part

        self.dxz_conv1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.dxz_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.dxz_conv3 = nn.Conv2d(512, 512, 1, 1, 0)
        self.dxz_conv4 = nn.Conv2d(512, 1, 1, 1, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         m.weight.data.normal_(0.0, std)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, std)
        #         m.bias.data.zero_()
        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x, z):
        # dx forward
        # print("dx input: ", x.size())
        dx = F.leaky_relu(self.dx_conv1_bn(self.dx_conv1(x)), 0.2)
        # print("dx shape 1: ", dx.size())
        dx = F.leaky_relu(self.dx_conv2_bn(self.dx_conv2(dx)), 0.2)
        # print("dx shape 2: ", dx.size())
        dx = F.leaky_relu(self.dx_conv3_bn(self.dx_conv3(dx)), 0.2)
        # print("dx shape 3: ", dx.size())
        dx = F.leaky_relu(self.dx_conv4_bn(self.dx_conv4(dx)), 0.2)
        # print("dx shape out: ", dx.size())

        # dz forward
        # print("dx input: ", z.size())
        dz = F.leaky_relu(self.dz_conv1(z), 0.2)
        # print("dz shape 1: ", dz.size())
        dz = F.leaky_relu(self.dz_conv2(dz), 0.2)
        # print("dz shape out : ", dz.size())

        # dxz forward
        xz = torch.cat((dx, dz), 1)
        # print("dxz input : ", xz.size())
        dxz = F.leaky_relu(self.dxz_conv1(xz), 0.2)
        # print("dxz input : ", dxz.size())
        dxz = F.leaky_relu(self.dxz_conv2(dxz), 0.2)
        dxz = F.leaky_relu(self.dxz_conv3(dxz), 0.2)
        # print("dxz input : ", dxz.size())
        dxz = F.sigmoid(self.dxz_conv4(dxz))
        # print("dxz out : ", dxz.size())

        return dxz


'''
shape Test
'''

if __name__ == "__main__":
    batch = 100
    z = torch.zeros([batch, z_dim])
    z = z.view([-1, z_dim, 1, 1])
    z = Variable(z)

    x = torch.zeros([batch, 1, 32, 32])
    x = Variable(x)

    # x2 = torch.zeros([batch, 32, 32, 1])
    # x2 = Variable(x2)

    p = P()
    q = Q()
    d = D()

    out = p.forward(z)
    out2 = q.forward(x)
    out3 = d.forward(x, z)

    r1 = out.data.numpy()
    r2 = out2.data.numpy()
    r3 = out3.data.numpy()

    print(r1.shape)
    print(r2.shape)
    print(r3.shape)
