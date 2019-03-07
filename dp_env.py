#!/usr/bin/env python3
import numpy as np

from mujoco.mocap import MocapDM
from mujoco.mujoco_interface import MujocoInterface

from gym.envs.mujoco import mujoco_env
from gym import utils

# TODO: load mocap data; calc rewards
# TODO: early stop

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        file_path = '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/dp_env_v1.xml'
        mujoco_env.MujocoEnv.__init__(self, file_path, 30)
        utils.EzPickle.__init__(self)
        self.mocap = MocapDM()
        self.interface = MujocoInterface()

        self.weight_pose = 0.5
        self.weight_vel = 0.05
        self.weight_end_eff = 0.15
        self.weight_root = 0.2
        self.weight_com = 0.1

        self.scale_pose = 2.0
        self.scale_vel = 0.1
        self.scale_end_eff = 40.0
        self.scale_root = 5.0
        self.scale_com = 10.0
        self.scale_err = 1.0

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        position = position[2:] # ignore x and y
        return np.concatenate((position, velocity))

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos[:]

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.dt = self.mocap.dt

    def calc_reward(self):
        assert len(self.mocap.data) != 0
        curr_time = env.get_time()
        idx_mocap = curr_time//self.dt

        target_mocap = self.mocap.data[idx_mocap, 1:]
        curr_pos = self.get_joint_configs()

        err_pos = self.interface.calc_pos_err(curr_pos, target_mocap)

        return 0.0

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward = self.calc_reward()
        done = False
        info = dict()
        return observation, reward, done, info

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        noise_low = -1e-2
        noise_high = 1e-2

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

if __name__ == "__main__":
    env = DPEnv()
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    print(action_size)
    while True:
        # ac[0] = (np.random.rand() - 0.5)*2
        ac[2] = 1
        env.step(ac)
        env.render()