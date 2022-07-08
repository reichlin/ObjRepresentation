import os
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from networks import phi, f, AE, f_network
import matplotlib.pyplot as plt

from torch.distributions.multivariate_normal import MultivariateNormal



class Representation:

    def __init__(self, N, a_dim, channels, args, device):
        super(Representation, self).__init__()

        self.device = device

        self.loss_type = args.loss_type
        self.frq = args.frq

        self.tau = args.tau
        self.m = args.m

        self.e2e = args.e2e

        self.gamma = args.gamma

        self.Psi = phi(N, channels, a_dim, 1, device).to(self.device)
        # self.f = f(a_dim, args.layers_f).to(self.device)
        self.optimizer = optim.Adam(self.Psi.parameters(), lr=1e-3)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def fit_Rep(self, batch, train=True):

        ot, ot1, at, pos, next_pos = batch[0], batch[1], batch[2], batch[3], batch[4]
        st, st1, at, pos, next_pos = ot.to(self.device), ot1.to(self.device), at.to(self.device), pos.to(self.device), next_pos.to(self.device)
        # pos_g = pos_g.to(self.device)

        # x = self.AE.get_latent(st)
        # x1 = self.AE.get_latent(st1)
        zt, ht = self.Psi(st)
        zt1, ht1 = self.Psi(st1)

        if self.e2e == 0:
            h_hat = self.f(ht[:, 0].detach())
            h1_hat = self.f(ht1[:, 0].detach())
        elif self.e2e == 1:
            h_hat = self.Psi.get_f(ht[:, 0]) #self.f(ht[:, 0])
            h1_hat = self.Psi.get_f(ht1[:, 0]) #self.f(ht1[:, 0])
        else:
            h_hat = ht[:, 0]
            h1_hat = ht1[:, 0]

        loss_Eqv = torch.mean(torch.sum(((zt1[:,0] - zt[:,0]) - at) ** 2, -1))
        avg_cos_sim = torch.mean(self.cos_sim((zt1[:,0] - zt[:,0]), at)).detach().cpu().item()

        if self.loss_type == 0:
            zht = torch.cat([zt[:, 0].detach(), ht[:, 0]], -1)#.detach()
            zht1 = torch.cat([zt1[:, 0].detach(), ht1[:, 0]], -1)

            pos_d = torch.sum((ht[:, 0].detach() - ht1[:, 0]) ** 2, -1) / self.tau

            rnd_idx = np.arange(ht.shape[0])
            np.random.shuffle(rnd_idx)
            neg_d = torch.sum((zht[rnd_idx].view(-1, 1, 4) - zht1) ** 2, -1) / self.tau

            loss_contrastive = - (-pos_d - torch.logsumexp(-neg_d, 0))
        else:
            zht = torch.cat([zt[:, 0].detach(), ht[:, 0]], -1)#.detach()
            zht1 = torch.cat([zt1[:, 0].detach(), ht1[:, 0]], -1)
            positive_d = torch.sum((zht - zht1) ** 2, -1)
            rnd_idx = np.arange(ht.shape[0])
            np.random.shuffle(rnd_idx)
            negative_dist = torch.clamp(self.m - torch.sum((zht[rnd_idx].view(-1, 4) - zht1[:]) ** 2, -1), min=0)
            negative_d = torch.sum(negative_dist, 0)
            loss_contrastive = torch.mean(positive_d) + torch.mean(negative_d)


        err_inv = torch.sum((ht[:, 0] - ht1[:, 0]) ** 2, -1)

        err_pos = (h_hat - zt[:, 0].detach()) ** 2
        pos_haptic = err_pos.sum(-1)

        err_neg = (h_hat - h1_hat) ** 2
        neg_haptic = err_neg.sum(-1)

        neg_idx = torch.argsort(err_inv)

        loss_haptic_pos = pos_haptic[neg_idx[int(pos_haptic.shape[0] * self.frq):]]
        loss_haptic_neg = neg_haptic[neg_idx[:int(pos_haptic.shape[0] * self.frq)]]
        loss_haptic = torch.mean(loss_haptic_pos) + torch.mean(loss_haptic_neg)

        frq_pos = torch.mean((pos_haptic < neg_haptic)*1.).detach().cpu().item()

        # if self.iter % 1 == 0:
        #     if frq_pos < self.ratio:
        #         self.gamma += 0.001
        #     else:
        #         self.gamma = max(self.gamma-0.001, 0)
        # self.iter += 1

        # o_hat = self.Psi.get_decoder(zt[:, 0].detach(), ht[:, 0].detach())
        # loss_dec = torch.mean(torch.sum((st - o_hat) ** 2, (1, 2, 3)))

        loss = loss_Eqv + torch.mean(loss_contrastive) + self.gamma*loss_haptic

        if train:

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # for j in range(1, 20):
        #     i = neg_idx[j].detach().cpu().item()
        #     print(torch.sum(means_kl_neg, (1, 2))[i])
        #     plt.imshow(np.transpose(ot[i].detach().cpu().numpy(), (1, 2, 0)))
        #     plt.show()

        # i = 2
        # plt.scatter(zt[i, 1].detach().cpu().numpy(), -zt[i, 0].detach().cpu().numpy())
        # plt.scatter(ht[i, 0, 0, 1].detach().cpu().numpy(), -ht[i, 0, 0, 0].detach().cpu().numpy())
        # plt.xlim((-70, 30))
        # plt.ylim((-40, 50))
        # plt.show()
        # plt.imshow(np.transpose(st[i].detach().cpu().numpy(), (1, 2, 0)))
        # plt.show()

        h_err = torch.mean(torch.abs((h1_hat - h_hat) - (next_pos[:, 1] - pos[:, 1]))).detach().cpu().numpy()
        h_pos_err = torch.mean(torch.abs(h_hat - pos[:,1])).detach().cpu().numpy()

        h_a = (h1_hat - h_hat)
        h_pos_diff = (next_pos[:, 1] - pos[:, 1])
        h_err_p = torch.mean(torch.sum(torch.abs(h_a - h_pos_diff), -1) * (torch.sum(torch.abs(h_pos_diff), -1) > 0))
        h_err_n = torch.mean(torch.sum(torch.abs(h_a - h_pos_diff), -1) * (torch.sum(torch.abs(h_pos_diff), -1) == 0))

        const_t = torch.nonzero(torch.sum(next_pos[:, 1] - pos[:, 1], -1))[:, 0].detach().cpu().numpy()

        avg_pos_haptic = torch.mean(pos_haptic).detach().cpu().numpy()
        avg_neg_haptic = torch.mean(neg_haptic).detach().cpu().numpy()

        err_inv_n = err_inv.detach().cpu().numpy()

        avg_contrastive = torch.mean(loss_contrastive).detach().cpu().item()

        avg_h_haptic_loss = loss_haptic.detach().cpu().item()
        # avg_reconstruction_loss = loss_dec.detach().cpu().item()
        avg_equivariance_loss = loss_Eqv.detach().cpu().item()

        # o_logs = {'ot': st[0].detach().cpu(),
        #           'ot_hat': o_hat[0].detach().cpu(),}
                  # 'ot_hat_real': o_hat_real[0].detach().cpu()}

        metrics_logs = {'eqv_loss': avg_equivariance_loss,
                        'cos_sim': avg_cos_sim,
                        'h_loss': avg_h_haptic_loss,
                        'pos_hapt': avg_pos_haptic,
                        'neg_hapt': avg_neg_haptic,
                        'avg_contrastive': avg_contrastive,
                        'frq_pos': frq_pos,
                        # 'dec_loss': avg_reconstruction_loss,
                        'h_dist': ht[:, 0].detach().cpu().numpy(),
                        'h_hat_dist': h_hat.detach().cpu().numpy(),
                        'z_dist': zt[:, 0].detach().cpu().numpy(),
                        'h_err': h_err,
                        'h_pos_err': h_pos_err,
                        'const_t': const_t,
                        'avg_err_inv': err_inv_n,
                        'h_err_p': h_err_p.detach().cpu().numpy(),
                        'h_err_n': h_err_n.detach().cpu().numpy()}

        return metrics_logs#, o_logs


    def get_rep(self, ot):

        zt, ht = self.Psi(ot)

        return zt.detach(), ht.detach()

    def save_model(self, epoch=None):
        if epoch is None:
            fname_psi = "./saved_models/Trained_Psi.mdl"
        else:
            fname_psi = "./saved_models/Psi_epoch="+str(epoch)+".mdl"
        torch.save(self.Psi.state_dict(), fname_psi)

    def load_model(self, epoch=None):
        if epoch is None:
            fname_psi = "./saved_models/Trained_Psi.mdl"
        else:
            fname_psi = "./saved_models/Psi_epoch="+str(epoch)+".mdl"
        state_dict_psi = torch.load(fname_psi, map_location=self.device)
        self.Psi.load_state_dict(state_dict_psi)
        self.Psi.eval()

    # def save_ae(self):
    #     fname_psi = "./saved_models/Trained_ae.mdl"
    #     torch.save(self.AE.state_dict(), fname_psi)
    #
    # def load_ae(self):
    #     fname_psi = "./saved_models/Trained_ae.mdl"
    #     state_dict_psi = torch.load(fname_psi, map_location=self.device)
    #     self.AE.load_state_dict(state_dict_psi)
    #     self.AE.eval()
















