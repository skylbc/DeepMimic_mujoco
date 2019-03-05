#!/usr/bin/env python3
import numpy as np

from mujoco.mocap import MocapDM
from mujoco.mujoco_interface import MujocoInterface

from gym.envs.mujoco import mujoco_env
from gym import utils

# TODO: load mocap data; calc rewards
# TODO: early stop

class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        file_path = '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/dp_env_v1.xml'
        mujoco_env.MujocoEnv.__init__(self, file_path, 30)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        position = position[2:] # ignore x and y
        return np.concatenate((position, velocity))

    def calc_reward(self):
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