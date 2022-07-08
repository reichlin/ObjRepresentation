import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time

class Simulator:

    def __init__(self):

        self.N = 100
        self.grid = np.zeros((3, self.N, self.N))
        # self.grip = None

        self.radius = 10
        self.img_grip = np.zeros((self.radius*2, self.radius*2))
        for i in range(-self.radius, self.radius):
            for j in range(-self.radius, self.radius):
                if np.abs(i) + np.abs(j) < self.radius:
                    self.img_grip[i + self.radius, j + self.radius] = 1

        self.img_obj = np.ones((self.radius*2, self.radius*2))

        self.tot_val = np.sum(self.img_grip) + np.sum(self.img_obj)

        self.max_a = 10
        self.vel = 5
        self.bouncing_coeff = 2.

        self.pos_grip = np.zeros(2)
        self.pos_obj = np.zeros(2)

    '''
        initial position of the gripper is always 0,0 for equivariance origin loss
    '''
    def reset(self, origin=True):

        if origin:
            self.pos_grip = np.zeros(2).astype(int) + self.radius
        else:
            self.pos_grip = np.random.randint(self.radius, high=self.N - self.radius, size=2)

        obj_placed = False
        while not obj_placed:
            self.pos_obj = np.random.randint(self.radius, high=self.N - self.radius, size=2)
            obj_placed = not self.check_touch()

        return

    def sample_random_action(self):
        at = self.max_a * (np.random.rand(2) * 2 - 1)
        while not self.action_valid(at):
            at = self.max_a * (np.random.rand(2) * 2 - 1)

        return at

    def check_touch(self):

        if self.pos_obj[0] == self.pos_grip[0] and self.pos_obj[1] == self.pos_grip[1]:
            return False
        return True

    def action_valid(self, a):

        if int(a[0])**2 + int(a[1])**2 == 0:
            return False

        if self.pos_grip[0] + a[0] < self.radius+2 or self.pos_grip[0] + a[0] > self.N-self.radius-2 or self.pos_grip[1] + a[1] < self.radius+2 or self.pos_grip[1] + a[1] > self.N-self.radius-2:
            return False

        return True

    def plot_obj(self):
        for i in range(-self.radius, self.radius, 1):
            for j in range(-self.radius, self.radius, 1):
                self.grid[1, self.pos_obj[0] + i, self.pos_obj[1] + j] += self.img_obj[i + self.radius, j + self.radius]

    def plot_grip(self):
        for i in range(-self.radius, self.radius, 1):
            for j in range(-self.radius, self.radius, 1):
                self.grid[0, self.pos_grip[0] + i, self.pos_grip[1] + j] += self.img_grip[i + self.radius, j + self.radius]

    def step(self, action):

        old_pox_x = self.pos_grip[0].copy()
        old_pos_y = self.pos_grip[1].copy()

        self.pos_grip[0] = np.clip(int(np.around(self.pos_grip[0] + action[0])), self.radius, self.N - self.radius)
        self.pos_grip[1] = np.clip(int(np.around(self.pos_grip[1] + action[1])), self.radius, self.N - self.radius)

        return np.array([self.pos_grip[0] - old_pox_x, self.pos_grip[1] - old_pos_y])


if __name__ == '__main__':

    N_trj = 100000
    N_steps = 2

    sim = Simulator()

    # img0 = np.zeros((N_trj, N_steps))
    a_t = np.zeros((N_trj, N_steps, 2))
    real_pos = np.zeros((N_trj, N_steps, 2, 2))

    tot_touch = 0
    tau = 0

    to_be_touched = True

    while tau < N_trj:

        tmp_touch = 0

        sim.reset(origin=False)

        for t in range(N_steps):

            touched = 0

            a = sim.sample_random_action()

            a = sim.step(a)

            a_t[tau, t] = a
            real_pos[tau, t, 0] = sim.pos_grip.copy()
            real_pos[tau, t, 1] = sim.pos_obj.copy()

            if not sim.check_touch():
                touched = 1
                obj_placed = False
                while not obj_placed:
                    sim.pos_obj = np.random.randint(sim.radius, high=sim.N - sim.radius, size=2)
                    obj_placed = sim.check_touch()

            tot_touch += touched
            tmp_touch += touched

        if to_be_touched:
            if tmp_touch > 0:
                tau += 1
                to_be_touched = False
        else:
            tau += 1
            to_be_touched = True

        print(tau)

    # np.save("./data_balanced/img0.npy", img0)
    np.save("./data_balanced/a_t.npy", a_t)
    np.save("./data_balanced/pos_t.npy", real_pos)

    print("tot interactions: " + str(tot_touch))

    print("full dataset collected")





