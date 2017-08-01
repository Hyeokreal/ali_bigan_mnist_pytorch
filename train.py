import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch
import torchvision
import os
from torch import optim
from torch.autograd import Variable
from itertools import chain
from model import P, Q, D

num_epochs = 20
batch_size = 100
z_dim = 100
x_dim = 32
sample_size = 100
glr = 0.000001
dlr = 0.000002
log_step = 1
sample_step = 200
sample_path = './samples'
model_path = './models'

if not os.path.isdir(sample_path):
    os.mkdir(sample_path)

if not os.path.isdir(model_path):
    os.mkdir(model_path)

img_size = 32
transform = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = dataset.MNIST(root='./data/',
                              train=True,
                              transform=transform,
                              download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

gx = P()
gz = Q()
dxz = D()

g_param = chain(gx.parameters(), gz.parameters())
d_param = dxz.parameters()

g_optimizer = optim.Adam(g_param, glr, betas=(0.5, 0.999))

d_optimizer = optim.Adam(d_param, glr, betas=(0.5, 0.999))


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def reset_grad():
    """Zero the gradient buffers."""
    gx.zero_grad()
    gz.zero_grad()
    dxz.zero_grad()


def denorm(x):
    """Convert range (-1, 1) to (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)


fixed_noise = to_variable(torch.randn(batch_size, z_dim))
fixed_noise = fixed_noise.view([-1, z_dim, 1, 1])
total_step = len(train_loader)

# ones_label = Variable(torch.ones(batch_size))
# zeros_label = Variable(torch.zeros(batch_size))

for epoch in range(num_epochs):
    gx.train(), gz.train()
    dxz.train()
    for i, (x, _) in enumerate(train_loader):

        x = to_variable(x)
        z = to_variable(torch.randn(batch_size, z_dim))
        z = z.view(-1, z_dim, 1, 1)

        x_hat = gx.forward(z)
        z_hat = gz.forward(x)

        d_enc = dxz.forward(x, z_hat)
        d_gen = dxz.forward(x_hat, z)

        d_enc_sum = torch.mean(d_enc)
        d_gen_sum = torch.mean(d_gen)

        d_loss = 0.5 * torch.mean(d_enc ** 2 + (1 - d_gen) ** 2)

        for p in gx.parameters():
            p.requires_grad = False
        for p in gz.parameters():
            p.requires_grad = False
        for p in dxz.parameters():
            p.requires_grad = True

        # dxz.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        z = to_variable(torch.randn(batch_size, z_dim))
        z = z.view(-1, z_dim, 1, 1)

        x_hat = gx.forward(z)
        z_hat = gz.forward(x)

        d_enc = dxz.forward(x, z_hat)
        d_gen = dxz.forward(x_hat, z)

        g_loss = 0.5 * torch.mean(d_gen ** 2 + (1 - d_enc) ** 2)

        for p in gx.parameters():
            p.requires_grad = True
        for p in gz.parameters():
            p.requires_grad = True
        for p in dxz.parameters():
            p.requires_grad = False

        # gx.zero_grad()
        # gz.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i) % log_step == 0:
            print(
                'Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, d_enc: %.4f, d_gen: %.4f '
                % (epoch + 1, num_epochs, i + 1, total_step,
                   g_loss.data[0], d_loss.data[0], d_enc_sum.data[0], d_gen_sum.data[0]))

        # save the sampled images
        if (i) % sample_step == 0:
            fake_images = gx.forward(fixed_noise)
            torchvision.utils.save_image(denorm(fake_images.data),
                                         os.path.join(sample_path,
                                                      'fake_samples-%d-%d.png' % (
                                                          epoch + 1, i + 1)))

        # save the sampled images recon
        if (i) % sample_step == 0:
            fake_z = gz.forward(x)
            fake_x = gx.forward(fake_z)
            torchvision.utils.save_image(denorm(fake_x.data),
                                         os.path.join(sample_path,
                                                      'recon_samples-%d-%d.png' % (
                                                          epoch + 1, i + 1)))

    gx_path = os.path.join(model_path, 'gx-%d.pkl' % (epoch + 1))
    gz_path = os.path.join(model_path, 'gz-%d.pkl' % (epoch + 1))
    dxz_path = os.path.join(model_path, 'dxz-%d.pkl' % (epoch + 1))
    torch.save(gx.state_dict(), gx_path)
    torch.save(gz.state_dict(), gz_path)
    torch.save(dxz.state_dict(), dxz_path)
