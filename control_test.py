#!/usr/bin/env python3
import numpy as np

from mujoco.mocap_v2 import MocapDM
from mujoco.mujoco_interface import MujocoInterface

from gym.envs.mujoco import mujoco_env
from gym import utils

class HumanoindDPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        file_path = '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/dp_env_v3.xml'
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
    import time

    chest_x = 0
    chest_y = 1
    chest_z = 2
    neck_x = 3
    neck_y = 4
    neck_z = 5

    right_shoulder_x = 6
    right_shoulder_y = 7
    right_shoulder_z = 8
    right_elbow = 9

    left_shoulder_x = 10
    left_shoulder_y = 11
    left_shoulder_z = 12
    left_elbow = 13

    right_hip_x = 14
    right_hip_y = 15
    right_hip_z = 16

    right_knee = 17
    right_ankle_x = 18
    right_ankle_y = 19
    right_ankle_z = 20

    left_hip_x = 21
    left_hip_y = 22
    left_hip_z = 23
    left_knee = 24
    left_ankle_x = 25
    left_ankle_y = 26
    left_ankle_z = 27

    torque = np.zeros(env.action_space.shape)
    torque[chest_y] = -1 
    torque[neck_y] = 1 
    # torque[right_hip_y] = -1
    # torque[left_elbow] = 0.3 
    # torque[left_knee] = -1 
    # torque[right_shoulder_y] = 1 
    # torque[left_shoulder_y] = 1 
    while True:
        env.step(torque)
        env.render()

    '''
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
    '''