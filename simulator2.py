import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time

class Simulator:

    def __init__(self):

        self.N = 64

        self.radius = 5
        self.img_grip = np.zeros((self.radius*2-1, self.radius*2-1))
        for i in range(-self.radius+1, self.radius):
            for j in range(-self.radius+1, self.radius):
                if np.abs(i) + np.abs(j) < self.radius:
                    self.img_grip[i + self.radius - 1, j + self.radius - 1] = 1

        self.img_obj = np.ones((self.radius*2-1, self.radius*2-1))

        self.tot_val = np.sum(self.img_grip) + np.sum(self.img_obj)

        self.pos_grip = np.zeros(2)
        self.pos_obj = np.zeros(2)

    def reset(self, counter, touch=True):

        pos_grip = np.random.randint(self.radius-1, high=self.N-self.radius+1, size=2)
        pos_grip_1 = np.random.randint(self.radius-1, high=self.N-self.radius+1, size=2)
        a = (pos_grip_1 - pos_grip)/self.N

        if touch:
            # pos_obj = pos_grip.copy()
            pos_obj = np.clip(pos_grip + np.random.randint(-self.radius, high=self.radius + 1, size=2), a_min=self.radius, a_max=self.N - self.radius - 1)
        else:
            pos_obj = np.random.randint(self.radius-1, high=self.N-self.radius+1, size=2)

        pos_obj_1 = pos_obj.copy()
        while self.check_touch(pos_grip, pos_obj_1):
            pos_obj_1 = np.random.randint(self.radius, high=self.N - self.radius, size=2)

        if np.sum(np.max(self.get_img(pos_grip, pos_obj), -1)) < 14:
            counter += 1

        return pos_grip, pos_grip_1, pos_obj, pos_obj_1, a, counter

    def check_touch(self, pos_grip, pos_obj):

        delta_x = np.abs(pos_obj[0]-pos_grip[0])
        delta_y = np.abs(pos_obj[1]-pos_grip[1])

        if delta_x + delta_y <= 2*self.radius-1:
            return True

        # if pos_obj[0] > pos_grip[0] - self.radius and pos_obj[0] < pos_grip[0] + self.radius:
        #     if pos_obj[1] > pos_grip[1] - self.radius and pos_obj[1] < pos_grip[1] + self.radius:
        #         return True
        return False

    def get_img(self, pos_grip, pos_obj):
        grid = np.zeros((3, self.N, self.N))
        for i in range(-self.radius+1, self.radius, 1):
            for j in range(-self.radius+1, self.radius, 1):
                grid[1, pos_obj[0] + i, pos_obj[1] + j] += self.img_obj[i + self.radius - 1, j + self.radius - 1]
                grid[0, pos_grip[0] + i, pos_grip[1] + j] += self.img_grip[i + self.radius - 1, j + self.radius - 1]
        return grid


if __name__ == '__main__':

    N_trj = 100000
    counter = 0

    sim = Simulator()

    a_t = np.zeros((N_trj, 2))
    imgs = np.zeros((N_trj, 3, sim.N, sim.N))
    next_imgs = np.zeros((N_trj, 3, sim.N, sim.N))
    real_pos = np.zeros((N_trj, 2, 2, 2))  # N_trj, time, grip/obj, (x,y)

    for trj in tqdm(range(N_trj)):

        if trj % 2 == 0:
            pos_grip, pos_grip_1, pos_obj, pos_obj_1, a, counter = sim.reset(counter, touch=True)
        else:
            pos_grip, pos_grip_1, pos_obj, pos_obj_1, a, counter = sim.reset(counter, touch=False)

        a_t[trj] = a
        # real_pos[trj, 0, 0] = pos_grip
        # real_pos[trj, 0, 1] = pos_obj
        # real_pos[trj, 1, 0] = pos_grip_1
        # real_pos[trj, 1, 1] = pos_obj_1
        imgs[trj] = sim.get_img(pos_grip, pos_obj)
        next_imgs[trj] = sim.get_img(pos_grip_1, pos_obj_1)

    np.save("./sim_dataset/data_giovanni/a.npy", a_t)
    np.save("./sim_dataset/data_giovanni/img.npy", imgs)
    np.save("./sim_dataset/data_giovanni/next_img.npy", next_imgs)

    print("full dataset collected")
    print("touch ratio: ", (counter*100./N_trj))





