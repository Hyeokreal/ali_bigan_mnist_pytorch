import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, nc, input_size, hparams, ngpu=1, leaky_slope=0.01, std=0.01):
        super(CNN, self).__init__()
        self.ngpu = ngpu  # num of gpu's to use
        self.leaky_slope = leaky_slope  # slope for leaky_relu activation
        self.std = std  # standard deviation for weights initialization
        self.input_size = input_size  # expected input size

        main = nn.Sequential()
        in_feat, num = nc, 0
        for op, k, s, p, out_feat, b, bn, dp, h in hparams:
            # add operation: conv2d or convTranspose2d
            if op == 'conv2d':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.conv'.format(num, in_feat, out_feat),
                    nn.Conv2d(in_feat, out_feat, k, s, padding=p, bias=b))
            elif op == 'convt2d':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.convt'.format(num, in_feat, out_feat),
                    nn.ConvTranspose2d(in_feat, out_feat, k, s, padding=p, bias=b))
            else:
                raise Exception('Not supported operation: {0}'.format(op))
            num += 1
            # add batch normalization layer
            if bn:
                main.add_module(
                    '{0}.pyramid.{1}-{2}.batchnorm'.format(num, in_feat, out_feat),
                    nn.BatchNorm2d(out_feat))
                num += 1
            # add dropout layer
            main.add_module(
                '{0}.pyramid.{1}-{2}.dropout'.format(num, in_feat, out_feat),
                nn.Dropout2d(p=dp))
            num += 1
            # add activation
            if h == 'leaky_relu':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.leaky_relu'.format(num, in_feat, out_feat),
                    nn.LeakyReLU(self.leaky_slope, inplace=True))
            elif h == 'sigmoid':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.sigmoid'.format(num, in_feat, out_feat),
                    nn.Sigmoid())
            elif h == 'tanh':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.tanh'.format(num, in_feat, out_feat),
                    nn.Tanh())
            elif h == 'linear':
                num -= 1  # 'Linear' do nothing
            else:
                raise Exception('Not supported activation: {0}'.format(h))
            num += 1
            in_feat = out_feat
        self.main = main

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, self.std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, self.std)
                m.bias.data.zero_()

    def forward(self, input):
        assert input.size(2) == self.input_size, \
            'Wrong input size: {0}. Expected {1}'.format(input.size(2),
                                                         self.input_size)
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)
        return output


def create_mnist_gx(nz=100, ngpu=0):
    hparams = [
        # op // kernel // strides // pad // fmaps // conv. bias // batch_norm // dropout //nonlinear
        ['convt2d', 4, 1, 0, 128, False, True, 0.0, 'leaky_relu'],
        ['convt2d', 4, 2, 1, 64, False, True, 0.0, 'leaky_relu'],
        ['convt2d', 4, 2, 1, 32, False, True, 0.0, 'leaky_relu'],
        ['convt2d', 4, 2, 1, 16, False, True, 0.0, 'leaky_relu'],
        # ['convt2d', 2, 1, 0, 32, True, True, 0.0, 'leaky_relu'],
        ['conv2d', 1, 1, 0, 1, False, False, 0.0, 'sigmoid'],
    ]
    return CNN(nz, 1, hparams, ngpu)


def create_mnist_gz(nz=100, ngpu=0):
    hparams = [
    # op // kernel // strides // pad // fmaps // conv. bias // batch_norm // dropout // nonlinear
        ['conv2d', 5, 1, 0, 32, False, True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 1, 0, 32, False, True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 2, 0, 64, False, True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 1, 0, 64, False, True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 1, 0, 128, False, True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 2, 0, 256, False, True, 0.0, 'leaky_relu'],
        # ['conv2d', 4, 2, 0, 256, False, True, 0.0, 'leaky_relu'],
        # ['conv2d', 4, 1, 0,  512, False, True, 0.0, 'leaky_relu'],
        # ['conv2d', 1, 1, 0,  512, False, True, 0.0, 'leaky_relu'],
        ['conv2d', 1, 1, 0, 2 * nz, True, False, 0.0, 'linear'],
    ]
    return CNN(1, 32, hparams, ngpu)




def create_mnist_dx(ngpu=0):
    hparams = [
        # op // kernel // strides // pad // fmaps // conv. bias // batch_norm // dropout //nonlinear
        ['conv2d', 5, 1, 0, 32, True, False, 0.2, 'leaky_relu'],
        ['conv2d', 4, 2, 0, 64, False, True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 2, 0, 128, False, True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 2, 0, 128, False, True, 0.2, 'leaky_relu'],
        # ['conv2d', 4, 1, 0, 256, False, True, 0.2, 'leaky_relu'],
    ]
    return CNN(1, 32, hparams, ngpu)


def create_mnist_dz(nz=100, ngpu=0):
    hparams = [
        # op // kernel // strides // pad // fmaps // conv. bias // batch_norm // dropout //nonlinear
        ['conv2d', 1, 1, 0, 128, False, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 0, 128, False, False, 0.2, 'leaky_relu'],
    ]
    return CNN(nz, 1, hparams, ngpu)


def create_mnist_dxz(ngpu=0):
    hparams = [
        # op // kernel // strides // pad // fmaps // conv. bias // batch_norm // dropout //nonlinear
        ['conv2d', 1, 1, 0, 256, True, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 0, 256, True, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 0, 1, True, False, 0.2, 'linear'],
    ]
    return CNN(256, 1, hparams, ngpu)


def create_models(nz, ngpu=1):
    gx = create_mnist_gx(nz, ngpu)
    gz = create_mnist_gz(nz, ngpu)
    dx = create_mnist_dx(ngpu)
    dz = create_mnist_dz(nz, ngpu)
    dxz = create_mnist_dxz(ngpu)
    return gx, gz, dx, dz, dxz


if __name__ == "__main__":
    batch = 100
    z_dim = 100

    z = torch.zeros([batch, z_dim])
    z = z.view([-1, z_dim, 1, 1])
    z = Variable(z)

    x = torch.zeros([batch, 1, 32, 32])
    x = Variable(x)

    gx, gz, dx, dz, dxz = create_models(z_dim, 0)

    gx_out = gx.forward(z)
    gz_out = gz.forward(x)

    dx_out = dx.forward(x)
    dz_out = dz.forward(z)

    dxz_in =torch.cat([dx_out, dz_out], 1)

    dxz_out = dxz.forward(dxz_in)

    print("gxout : ", gx_out.size())
    print("gzout : ", gz_out.size())
    print("dxout : ", dx_out.size())
    print("dzout : ", dz_out.size())
    print("dxzout : ", dxz_out.size())
