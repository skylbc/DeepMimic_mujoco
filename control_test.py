#!/usr/bin/env python3
import numpy as np

from mujoco.mocap import MocapDM
from mujoco.mujoco_interface import MujocoInterface

from gym.envs.mujoco import mujoco_env
from gym import utils

class HumanoindDPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        file_path = '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/dp_env_v1.xml'
        mujoco_env.MujocoEnv.__init__(self, file_path, 30)
        utils.EzPickle.__init__(self)

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos[:]
    
    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        return self.get_joint_configs(), 0.0, False, dict()

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self.get_joint_configs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


if __name__ == "__main__":
    env = HumanoindDPEnv()

    mocap_filepath = '/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt'

    mocap = MocapDM()
    mocap.load_mocap(mocap_filepath)

    interface = MujocoInterface()
    interface.init(env.sim, mocap.dt)

    import time

    round_idx = 0

    update_time_int = 1.0/30
    mocap_time_int = 0.0625

    delta = 0.1 * 0.09

    idx_mocap = 0

    target_pos = mocap.data[0, 1:]
    curr_pos = env.get_joint_configs()
    err_pos = interface.calc_pos_err(curr_pos, target_pos)
    torque = err_pos
    target_pos = mocap.data[0, 1:]

    # idx = 3
    # torque = np.zeros(env.action_space.shape)
    # torque[idx] = (np.random.rand() - 0.5) * 10

    curr_time = 0.0

    # idx += 1
    # if idx >= env.action_space.shape[0]:
    #     idx = 0
    # torque = np.zeros(env.action_space.shape)
    # torque[idx] = 1.0
    torque = np.ones(env.action_space.shape)

    while True:
        curr_time = env.get_time()
        # curr_time += 0.09

        if curr_time // update_time_int != (curr_time - delta) // update_time_int:
            # update mujoco
            # env.goto(target_pos)
            pass

        if curr_time // mocap_time_int != (curr_time - delta) // mocap_time_int:
            # update mocap
            print('update mocap', idx_mocap)
            # torque = np.zeros(env.action_space.shape)
            # torque[idx] = (np.random.rand() - 0.5) * 10
            idx_mocap += 1
            idx_mocap = idx_mocap % len(mocap.data)
            target_pos = mocap.data[idx_mocap, 1:]
            curr_pos = env.get_joint_configs()

            err_pos = interface.calc_pos_err(curr_pos, target_pos)
            # torque = np.random.rand(np.shape(err_pos)[0])
            # torque = err_pos * 2
            torque = -torque

        env.step(torque)
        env.render()

    # interface = MujocoInterface()
    # interface.init(env.sim, mocap.dt)