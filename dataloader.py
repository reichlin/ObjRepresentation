import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class DatasetSim(Dataset):

    def __init__(self, frq=0.5, N=28, batch_size=32):

        # self.N = 100
        # self.radius = 10
        # self.grid = np.zeros((3, self.N, self.N))
        #
        # self.img_grip = np.zeros((self.radius * 2, self.radius * 2))
        # for i in range(-self.radius, self.radius):
        #     for j in range(-self.radius, self.radius):
        #         if np.abs(i) + np.abs(j) < self.radius:
        #             self.img_grip[i + self.radius, j + self.radius] = 1
        #
        # self.img_obj = np.ones((self.radius * 2, self.radius * 2))

        self.N = N
        self.radius = 2
        self.touch_level = 2*self.radius-1
        self.batch_size = batch_size
        self.frq = frq

        self.img_grip = np.zeros((self.radius * 2 - 1, self.radius * 2 - 1))
        for i in range(-self.radius + 1, self.radius):
            for j in range(-self.radius + 1, self.radius):
                if np.abs(i) + np.abs(j) < self.radius:
                    self.img_grip[i + self.radius - 1, j + self.radius - 1] = 1

        self.img_obj = np.ones((self.radius * 2 - 1, self.radius * 2 - 1))

        # tmp_a = np.load(data_folder + "/a_t.npy", allow_pickle=True)
        # tmp_pos = np.load(data_folder + "/pos_t.npy", allow_pickle=True)
        #
        # self.pos = np.concatenate([tmp_pos[i, 0:1].astype(int) for i in range(tmp_pos.shape[0])])
        # self.next_pos = np.concatenate([tmp_pos[i, 1:2].astype(int) for i in range(tmp_pos.shape[0])])
        # self.a = np.concatenate([tmp_a[i:i+1] for i in range(tmp_a.shape[0])])

    def __len__(self):
        return self.batch_size*10

    def __getitem__(self, idx):

        pos_grip, pos_grip_1, pos_obj, pos_obj_1, a = self.gen_interaction()

        img = torch.from_numpy(self.get_img(pos_grip, pos_obj)).float()
        next_img = torch.from_numpy(self.get_img(pos_grip_1, pos_obj_1)).float()
        a = torch.from_numpy(a).float() / self.N
        real_pos = torch.from_numpy(np.concatenate([np.expand_dims(pos_grip, 0), np.expand_dims(pos_obj, 0)])).float() / self.N
        next_real_pos = torch.from_numpy(np.concatenate([np.expand_dims(pos_grip_1, 0), np.expand_dims(pos_obj_1, 0)])).float() / self.N

        return img, next_img, a, real_pos, next_real_pos

    def get_ladder(self):

        imgs = np.zeros((self.N-2, 3, self.N, self.N))
        for i in range(1, self.N-1):
            pos_grip = np.ones(2)*i
            pos_obj = np.random.randint(self.radius - 1, high=self.N - self.radius + 1, size=2)
            imgs[i-1] = self.get_img(pos_grip, pos_obj)
        return torch.from_numpy(imgs).float()

    def get_grid(self):

        imgs = np.zeros(((self.N-2)**2, 3, self.N, self.N))
        for i in range(1, self.N-1):
            for j in range(1, self.N - 1):
                pos_grip = np.array([2., 2.])
                pos_obj = np.array([float(i), float(j)])
                imgs[(i-1)*(self.N-2)+(j-1)] = self.get_img(pos_grip, pos_obj)
        return torch.from_numpy(imgs).float()

    def gen_interaction(self):

        pos_grip = np.random.randint(self.radius - 1, high=self.N - self.radius + 1, size=2)
        pos_grip_1 = np.random.randint(self.radius - 1, high=self.N - self.radius + 1, size=2)
        a = pos_grip_1 - pos_grip

        if np.random.random_sample() < self.frq:  # tocco
            pos_obj = np.clip(pos_grip + np.random.randint(-self.radius, high=self.radius + 1, size=2), a_min=self.radius, a_max=self.N - self.radius - 1)
            pos_obj_1 = pos_obj.copy()
            while self.check_touch(pos_grip, pos_obj_1):
                pos_obj_1 = np.random.randint(self.radius - 1, high=self.N - self.radius + 1, size=2)
        else:  # non tocco
            pos_obj = np.random.randint(self.radius - 1, high=self.N - self.radius + 1, size=2)
            # TODO: uncomment below if touching changes position of the object strictly
            # while self.check_touch(pos_grip, pos_obj):
            #     pos_obj = np.random.randint(self.radius - 1, high=self.N - self.radius + 1, size=2)
            pos_obj_1 = pos_obj.copy()

        return pos_grip, pos_grip_1, pos_obj, pos_obj_1, a

    def check_touch(self, pos_grip, pos_obj):

        delta_x = np.abs(pos_obj[0]-pos_grip[0])
        delta_y = np.abs(pos_obj[1]-pos_grip[1])

        if delta_x + delta_y <= self.touch_level:
            return True

    def get_img(self, grip_pos, obj_pos):

        img = np.zeros((3, self.N, self.N))
        img = self.plot_grip(img, grip_pos)
        img = self.plot_obj(img, obj_pos)
        return img.copy()

    def plot_obj(self, img, pos):
        for i in range(-self.radius+1, self.radius, 1):
            for j in range(-self.radius+1, self.radius, 1):
                img[1, int(pos[0]) + i, int(pos[1]) + j] += self.img_obj[i + self.radius - 1, j + self.radius - 1]
        return img

    def plot_grip(self, img, pos):
        for i in range(-self.radius+1, self.radius, 1):
            for j in range(-self.radius+1, self.radius, 1):
                img[0, int(pos[0]) + i, int(pos[1]) + j] += self.img_grip[i + self.radius - 1, j + self.radius - 1]
        return img

    # def plot_obj(self, img, pos):
    #     for i in range(-self.radius, self.radius, 1):
    #         for j in range(-self.radius, self.radius, 1):
    #             img[1, int(pos[0]) + i, int(pos[1]) + j] += self.img_obj[i + self.radius, j + self.radius]
    #     return img
    #
    # def plot_grip(self, img, pos):
    #     for i in range(-self.radius, self.radius, 1):
    #         for j in range(-self.radius, self.radius, 1):
    #             img[0, int(pos[0]) + i, int(pos[1]) + j] += self.img_grip[i + self.radius, j + self.radius]
    #     return img


if __name__ == '__main__':

    dataloader = DatasetSim("sim_dataset/data_balanced")
    batch = dataloader.__getitem__(100)