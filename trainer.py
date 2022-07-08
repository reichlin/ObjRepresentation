import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse

from representation import Representation
from dataloader import DatasetSim

# import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def train_rep(dataloader, N, args, device, writer):

    # print_frq = 20
    EPOCHS = 100000
    a_dim = 2
    channels = 3

    datasim = DatasetSim(frq=args.frq, N=N, batch_size=batch_size)

    Rep_learner = Representation(N, a_dim, channels, args, device)

    # fname_psi = "./saved_models/hinge2.mdl"
    # state_dict_psi = torch.load(fname_psi, map_location=device)
    # Rep_learner.Psi.load_state_dict(state_dict_psi)


    log_counter = 0
    for e in tqdm(range(EPOCHS)):

        # if e % 200 < 100:
        #     Rep_learner.gamma *= 1.05
        # else:
        #     Rep_learner.gamma *= 0.95

        Rep_learner.Psi.train()

        avg_equivariance_loss = 0
        avg_cos_sim = 0
        avg_h_loss = 0
        # avg_reconstruction_loss = 0
        avg_pos_hapt = 0
        avg_neg_hapt = 0
        avg_frq_pos = 0
        avg_h_err = 0
        avg_err_inv = 0
        avg_contrastive = 0
        h_pos_err = 0

        h_err_p = 0
        h_err_n = 0

        x_min = 1000
        y_min = 1000

        len_dataset = len(dataloader)
        for i, batch in enumerate(dataloader):

            metrics_logs = Rep_learner.fit_Rep(batch)

            if x_min > np.min(metrics_logs['z_dist'][:, 0]):
                x_min = np.min(metrics_logs['z_dist'][:, 0])
            if y_min > np.min(metrics_logs['z_dist'][:, 1]):
                y_min = np.min(metrics_logs['z_dist'][:, 1])

            avg_equivariance_loss += metrics_logs['eqv_loss']
            avg_cos_sim += metrics_logs['cos_sim']
            avg_h_loss += metrics_logs['h_loss']
            # avg_reconstruction_loss += metrics_logs['dec_loss']
            avg_pos_hapt += metrics_logs['pos_hapt']
            avg_neg_hapt += metrics_logs['neg_hapt']
            avg_frq_pos += metrics_logs['frq_pos']
            avg_h_err += metrics_logs['h_err']
            avg_err_inv += np.mean(metrics_logs['avg_err_inv'])
            avg_contrastive += metrics_logs['avg_contrastive']
            h_pos_err += metrics_logs['h_pos_err']

            h_err_p = metrics_logs['h_err_p']
            h_err_n = metrics_logs['h_err_n']

            # if i % print_frq == print_frq-1:
        writer.add_scalar("Training/avg_equivariance_loss", avg_equivariance_loss/len_dataset, log_counter)
        writer.add_scalar("Training/avg_cos_sim", avg_cos_sim/len_dataset, log_counter)
        writer.add_scalar("Training/avg_h_loss", avg_h_loss/len_dataset, log_counter)
        writer.add_scalar("Training/avg_pos_hapt", avg_pos_hapt/len_dataset, log_counter)
        writer.add_scalar("Training/avg_neg_hapt", avg_neg_hapt/len_dataset, log_counter)
        writer.add_scalar("Training/avg_frq_pos", avg_frq_pos/len_dataset, log_counter)
        # writer.add_scalar("Training/avg_reconstruction_loss", avg_reconstruction_loss/len_dataset, log_counter)
        writer.add_scalar("Training/avg_h_err", avg_h_err/len_dataset, log_counter)
        writer.add_scalar("Training/avg_err_inv", avg_err_inv/len_dataset, log_counter)
        writer.add_scalar("Training/avg_contrastive", avg_contrastive/len_dataset, log_counter)
        writer.add_scalar("Training/h_pos_err", h_pos_err / len_dataset, log_counter)

        writer.add_scalar("Training/h_err_p", h_err_p / len_dataset, log_counter)
        writer.add_scalar("Training/h_err_n", h_err_n / len_dataset, log_counter)

        # writer.add_image("ot", o_logs['ot'], log_counter, dataformats='CHW')
        # writer.add_image("ot_hat", torch.clamp(o_logs['ot_hat'], 0., 1.), log_counter, dataformats='CHW')
        # writer.add_image("ot_hat_real", torch.clamp(o_logs['ot_hat_real'], 0., 1.), log_counter, dataformats='CHW')
        fig = plt.figure()
        plt.scatter(metrics_logs['h_dist'][:, 0], metrics_logs['h_dist'][:, 1])
        plt.scatter(metrics_logs['h_dist'][metrics_logs['const_t'], 0], metrics_logs['h_dist'][metrics_logs['const_t'], 1])
        writer.add_figure("h_dist", fig, log_counter)
        fig2 = plt.figure()
        plt.scatter(metrics_logs['z_dist'][:, 0], metrics_logs['z_dist'][:, 1])
        writer.add_figure("z_dist", fig2, log_counter)
        fig3 = plt.figure()
        plt.hist(metrics_logs['h_dist'][:, 0], 100, alpha=0.5)
        plt.hist(metrics_logs['h_dist'][:, 1], 100, alpha=0.5)
        writer.add_figure("h_hist", fig3, log_counter)
        fig5 = plt.figure()
        plt.scatter(metrics_logs['h_hat_dist'][:, 0], metrics_logs['h_hat_dist'][:, 1])
        plt.scatter(metrics_logs['h_hat_dist'][metrics_logs['const_t'], 0], metrics_logs['h_hat_dist'][metrics_logs['const_t'], 1])
        writer.add_figure("h_hat_dist", fig5, log_counter)

        fig6 = plt.figure()
        non_const_t = np.arange(metrics_logs['avg_err_inv'].shape[0])
        j = 0
        for i in range(metrics_logs['avg_err_inv'].shape[0]):
            if i in metrics_logs['const_t']:
                non_const_t = np.delete(non_const_t, j)
            else:
                j += 1
        err_inv_n = metrics_logs['avg_err_inv'][non_const_t]
        err_inv_p = metrics_logs['avg_err_inv'][metrics_logs['const_t']]
        plt.hist(np.log(err_inv_n[np.nonzero(err_inv_n)]), 100, alpha=0.5)
        plt.hist(np.log(err_inv_p[np.nonzero(err_inv_p)]), 100, alpha=0.5)
        writer.add_figure("err_inv_dist", fig6, log_counter)

        high_idx = np.argsort(metrics_logs['avg_err_inv'])[:int(batch_size * args.frq)]
        sum = 0.
        for idx in high_idx:
            sum = sum + 1. if idx in non_const_t else sum + 0.
        writer.add_scalar("Training/prop_p_top_frq", sum/len(high_idx), log_counter)

        imgs = datasim.get_ladder()
        imgs = imgs.to(device)
        # x = Rep_learner.AE.get_latent(imgs)
        z, h = Rep_learner.Psi(imgs)
        fig4 = plt.figure()
        plt.plot(np.arange(z.shape[0])/z.shape[0], z[:,0,0].detach().cpu().numpy())
        plt.plot(np.arange(z.shape[0])/z.shape[0], z[:,0,1].detach().cpu().numpy())
        writer.add_figure("z_ladder", fig4, log_counter)

        # imgs = np.zeros((int((datasim.N - 2)/3.) ** 2, 3, datasim.N, datasim.N))
        # for i in range(1, datasim.N - 1, 3):
        #     for j in range(1, datasim.N - 1, 3):
        #         pos_grip = np.array([25., 25.])
        #         pos_obj = np.array([float(i), float(j)])
        #         imgs[(i - 1) * (datasim.N - 2) + (j - 1)] = datasim.get_img(pos_grip, pos_obj)
        # imgs = torch.from_numpy(imgs).float().to(device)
        # z, h = Rep_learner.Psi(imgs)
        # fig7 = plt.figure()
        # plt.scatter(h[:, 0, 0].detach().cpu().numpy(), h[:, 0, 1].detach().cpu().numpy())
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        # writer.add_figure("h_grid", fig7, log_counter)
        #
        # imgs = np.zeros((int((datasim.N - 2)/3) ** 2, 3, datasim.N, datasim.N))
        # for i in range(1, datasim.N - 1, 3):
        #     for j in range(1, datasim.N - 1, 3):
        #         pos_grip = np.array([float(i), float(j)])
        #         pos_obj = np.array([25., 25.])
        #         imgs[(i - 1) * (datasim.N - 2) + (j - 1)] = datasim.get_img(pos_grip, pos_obj)
        # imgs = torch.from_numpy(imgs).float().to(device)
        # z, h = Rep_learner.Psi(imgs)
        # fig10 = plt.figure()
        # plt.scatter(h[:, 0, 0].detach().cpu().numpy(), h[:, 0, 1].detach().cpu().numpy())
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        # writer.add_figure("z_grid", fig10, log_counter)

        # imgs = datasim.get_grid()
        # imgs = imgs.to(device)
        # z, h = Rep_learner.Psi(imgs)
        # h = Rep_learner.f(h[:, 0].detach())
        # fig7 = plt.figure()
        # plt.scatter(h[:, 0].detach().cpu().numpy(), h[:, 1].detach().cpu().numpy())
        # writer.add_figure("h_grid", fig7, log_counter)

        # i = 2
        # plt.scatter(metrics_logs['z_dist'][0, 1].detach().cpu().numpy(), -metrics_logs['z_dist'][0, 0].detach().cpu().numpy())
        # plt.scatter(metrics_logs['h_dist'][0, 1].detach().cpu().numpy(), -metrics_logs['h_dist'][0, 0].detach().cpu().numpy())
        # plt.xlim((-70, 30))
        # plt.ylim((-40, 50))
        # plt.show()
        # plt.imshow(np.transpose(st[i].detach().cpu().numpy(), (1, 2, 0)))
        # plt.show()

        img = batch[0][0]
        z_x = np.clip(int((metrics_logs['z_dist'][0, 0] - x_min)*N), a_min=0, a_max=N-1) #np.clip(int(metrics_logs['z_dist'][0, 0] - x_min), a_min=0, a_max=27)
        z_y = np.clip(int((metrics_logs['z_dist'][0, 1] - y_min)*N), a_min=0, a_max=N-1) #np.clip(int(metrics_logs['z_dist'][0, 1] - y_min), a_min=0, a_max=27)
        img[2, z_x, z_y] = 1
        h_x = np.clip(int((metrics_logs['h_hat_dist'][0, 0] - x_min)*N), a_min=0, a_max=N-1) #np.clip(int(metrics_logs['h_dist'][0, 0] - x_min), a_min=0, a_max=27)
        h_y = np.clip(int((metrics_logs['h_hat_dist'][0, 1] - y_min)*N), a_min=0, a_max=N-1) #np.clip(int(metrics_logs['h_dist'][0, 1] - y_min), a_min=0, a_max=27)
        img[1:3, h_x, h_y] = 1
        writer.add_image("estimated_pos", img, log_counter, dataformats='CHW')

        log_counter += 1

        # if e % 1000 == 999:
        #     Rep_learner.save_model(e)

    return Rep_learner


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_type', default=0, type=int, help="contrastive loss type, 0: infoNCE, 1: hinge")
    parser.add_argument('--frq', default=0.5, type=float, help="frq")

    parser.add_argument('--tau', default=0.1, type=float, help="infoNCE temperature")
    parser.add_argument('--m', default=0.01, type=float, help="hinge const")

    parser.add_argument('--e2e', default=2, type=int, help="0: detach f, 1: e2e f, 2: no f")
    parser.add_argument('--layers_f', default=4, type=int, help="number of layers f")

    parser.add_argument('--gamma', default=10., type=float, help="f reg")

    parser.add_argument('--seed', default=0, type=int, help="seed")
    args = parser.parse_args()

    frq = 0.5
    batch_size = 256
    N = 100

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloader = DataLoader(DatasetSim(frq=frq, N=N, batch_size=batch_size), batch_size=batch_size, shuffle=True, num_workers=0)

    name = "l_type="+str(args.loss_type)
    name += "_frq="+str(args.frq)
    if args.loss_type == 0:
        name += "_tau=" + str(args.tau)
    else:
        name += "_m=" + str(args.m)
    if args.e2e == 0:
        name += "_detach_f_"
        name += "_layers_f=" + str(args.layers_f)
    elif args.e2e == 1:
        name += "_f_layers_f=" + str(args.layers_f)
        name += "_gamma=" + str(args.gamma)
    else:
        name += "_no_f_gamma=" + str(args.gamma)
    name += "_seed=" + str(args.seed)

    writer = SummaryWriter("./logs_balanced/"+name+"big_o4_3_stable5_correct_num3")
    rep = train_rep(dataloader, N, args, device, writer)

    writer.close()
