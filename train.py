import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch
import torchvision
import os
from torch import optim
from torch.autograd import Variable
from model import P, Q, D

num_epochs = 20
batch_size = 100
z_dim = 100
x_dim = 32
sample_size = 100
glr = 0.00001
dlr = 0.00002
log_step = 1
sample_step = 50
sample_path = './samples'
model_path = './models'

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

g_optimizer = optim.Adam(list(gx.parameters()) + list(gz.parameters()), glr, betas=(0.5, 0.999))

d_optimizer = optim.Adam(dxz.parameters(), glr, betas=(0.5, 0.999))


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


'''
def sample():
    g_path = os.path.join(model_path, 'gxz-%d.pkl' % (num_epochs))
    d_path = os.path.join(model_path, 'discriminator-%d.pkl' % (num_epochs))
    generator.load_state_dict(torch.load(g_path))
    discriminator.load_state_dict(torch.load(d_path))
    generator.eval()
    discriminator.eval()

    # Sample the images
    noise = to_variable(torch.randn(sample_size, z_dim))
    fake_images = generator(noise)
    sample = os.path.join(sample_path, 'fake_samples-final.png')
    torchvision.utils.save_image(denorm(fake_images.data), sample, nrow=12)
    print("Saved sampled images to '%s'" % sample)
'''

fixed_noise = to_variable(torch.randn(batch_size, z_dim))
fixed_noise = fixed_noise.view([-1, z_dim, 1, 1])
total_step = len(train_loader)

# ones_label = Variable(torch.ones(batch_size))
# zeros_label = Variable(torch.zeros(batch_size))

for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_loader):

        x = to_variable(x)
        z = to_variable(torch.randn(batch_size, z_dim))
        z = z.view(-1, z_dim, 1, 1)

        x_hat = gx.forward(z)
        z_hat = gz.forward(x)

        d_enc = dxz.forward(x, z_hat)
        d_gen = dxz.forward(x_hat, z)

        d_loss = 0.5 * torch.mean(d_enc ** 2 + (1 - d_gen) ** 2)
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        z = to_variable(torch.randn(batch_size, z_dim))
        z = z.view(-1, z_dim, 1, 1)

        x_hat = gx.forward(z)
        z_hat = gz.forward(x)

        d_enc = dxz.forward(x, z_hat)
        d_gen = dxz.forward(x_hat, z)

        g_loss = 0.5 * torch.mean(d_gen ** 2 + (1 - d_enc) ** 2)
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i) % log_step == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f '
                  % (epoch + 1, num_epochs, i + 1, total_step,
                     g_loss.data[0], d_loss.data[0]))

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
