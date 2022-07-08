import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, in_dim, out_dims):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 64)

        self.module1 = nn.Sequential(nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU())

        self.module2 = nn.Sequential(nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU())

        self.module3 = nn.Sequential(nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, out_dims))

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = self.module1(z) + z
        z = self.module2(z) + z
        return self.module3(z)


class Base_Encoder(nn.Module):

    def __init__(self, N, channels, out_dims):
        super().__init__()

        if N == 28:
            self.dim_f = [64, 13, 13]
            self.encoder_cnn = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(64, self.dim_f[0], kernel_size=3, stride=2, padding=0),
                                             nn.ReLU())
        else:
            self.dim_f = [64, 24, 24]
            self.encoder_cnn = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
                                             nn.ReLU(),)
                                             # nn.Conv2d(64, self.dim_f[0], kernel_size=3, stride=2, padding=0),
                                             # nn.ReLU())

        self.encoder = nn.Sequential(nn.Flatten(),
                                     nn.Linear(self.dim_f[0] * self.dim_f[1] * self.dim_f[2], 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, out_dims))

    def forward(self, x):
        z = self.encoder_cnn(x)
        return self.encoder(z)


class Base_Decoder(nn.Module):

    def __init__(self, N, in_dim, channels):
        super().__init__()

        if N == 28:
            self.dim_f = [32, 6, 6]
            self.fc = nn.Sequential(nn.Linear(in_dim, self.dim_f[0] * self.dim_f[1] * self.dim_f[2]))
            self.decoder = nn.Sequential(nn.ConvTranspose2d(self.dim_f[0], 64, 3, stride=2, padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(64, 64, 4, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(64, channels, 3, stride=1, padding=1))
        else:
            # TODO
            pass

    def forward(self, x):
        x = self.fc(x) #F.relu(self.fc(x))
        x = x.view(-1, self.dim_f[0], self.dim_f[1], self.dim_f[2])
        return self.decoder(x)


# class Base_Encoder(nn.Module):
#
#     def __init__(self, channels, out_dims):
#         super().__init__()
#
#         # self.dim_f = [8, 6, 6]
#         self.dim_f = [32, 2, 2]
#
#         self.encoder = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.Conv2d(64, self.dim_f[0], kernel_size=3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.Flatten(),
#                                      nn.Linear(self.dim_f[0] * self.dim_f[1] * self.dim_f[2], 64),
#                                      nn.ReLU(),
#                                      nn.Linear(64, out_dims))
#
#     def forward(self, x):
#         return self.encoder(x)
#
#
# class Base_Decoder(nn.Module):
#
#     def __init__(self, in_dim, channels):
#         super().__init__()
#
#         # self.dim_f = [8, 6, 6]
#         # k = 3
#         self.dim_f = [32, 2, 2]
#         k = 5
#
#         self.fc = nn.Linear(in_dim, self.dim_f[0] * self.dim_f[1] * self.dim_f[2])
#
#         self.decoder = nn.Sequential(nn.ConvTranspose2d(self.dim_f[0], 64, 3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.ConvTranspose2d(64, 64, k, stride=2, padding=0),
#                                      nn.ReLU(),
#                                      nn.ConvTranspose2d(64, channels, 4, stride=2, padding=0))
#
#     def forward(self, x):
#         x = F.relu(self.fc(x))
#         x = x.view(-1, self.dim_f[0], self.dim_f[1], self.dim_f[2])
#         return self.decoder(x)


'''
    decoder solo sulle medie
'''
class phi(nn.Module):

    def __init__(self, N, channels, a_dim, n_objs, device):
        super().__init__()

        self.a_dim = a_dim
        self.n_objs = n_objs
        self.device = device

        self.encoder_z = Base_Encoder(N, channels, a_dim*2)
        self.encoder_h = Base_Encoder(N, channels, a_dim*2)

        # self.decoder = Base_Decoder(N, a_dim+a_dim, channels)
        # self.decoder_h = Base_Decoder(a_dim, 1)

        # self.f = nn.Sequential(nn.Linear(a_dim, 64),
        #                        nn.ReLU(),
        #                        nn.Linear(64, 64),
        #                        nn.ReLU(),
        #                        nn.Linear(64, a_dim))

    def forward(self, o):
        z = torch.sigmoid(self.encoder_z(o).view(-1, 2, self.a_dim))
        h = torch.sigmoid(self.encoder_h(o).view(-1, 2, self.a_dim))
        # h *= self.mask.detach()
        return z, h

    def get_decoder(self, z, h):
        # return self.decoder_z(z), self.decoder_h(h)
        x = torch.cat([z, h], -1)
        return self.decoder(x)

    def get_f(self, h):
        return self.f(h)


class f(nn.Module):

    def __init__(self, a_dim, n_layers):
        super().__init__()

        modules = []
        modules.append(nn.Linear(a_dim, 64))
        modules.append(nn.ReLU())
        for _ in range(n_layers):
            modules.append(nn.Linear(64, 64))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(64, a_dim))

        self.f = nn.Sequential(*modules)

    def forward(self, h):
        # x = torch.cat([z, h], -1)
        return self.f(h)


class AE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Base_Encoder(3, 8)
        self.decoder = Base_Decoder(8, 3)

    def get_latent(self, o):
        return self.encoder(o)

    def get_o(self, x):
        return self.decoder(x)



class Block2(nn.Module):

    def __init__(self, in_dim, out_dims):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 64)

        self.module1 = nn.Sequential(nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU())

        self.module2 = nn.Sequential(nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, out_dims))

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = self.module1(z) + z
        return self.module2(z)


class f_network(nn.Module):

    def __init__(self, a_dim, n_layers):
        super().__init__()

        self.blk1 = Block2(a_dim, 64)
        self.blk2 = Block2(64, 64)
        self.blk3 = Block2(64, a_dim)

    def forward(self, h):
        z = self.blk1(h)
        z = self.blk2(z) + z
        return self.blk3(z)

