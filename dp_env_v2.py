#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd

from mujoco.mocap import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from gym.envs.mujoco import mujoco_env
from gym import utils

from config import Config
from pyquaternion import Quaternion

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

def quat2euler(quat):
    elements = quat.elements
    q0, q1, q2, q3 = elements[0], elements[1], elements[2], elements[3]
    phi = math.atan2(2.0*(q0*q1+q2*q3), 1.0-2.0*(q1*q1+q2*q2))
    theta = math.asin(2.0*(q0*q2-q3*q1))
    psi = math.atan2(2.0*(q0*q3+q1*q2), 1.0-2.0*(q2*q2+q3*q3))
    return phi, theta, psi

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

def degree2quaternion(theta_x, theta_y, theta_z):
    rot_x = np.array([[ 1,         0,                0          ], 
                      [ 0, math.cos(theta_x), -math.sin(theta_x)], 
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)], 
                      [        0,          1,       0        ], 
                      [-math.sin(theta_y), 0,  math.cos(theta_y)]])

    rot_z = np.array([[math.cos(theta_z), -math.sin(theta_z), 0 ], 
                      [math.sin(theta_z),  math.cos(theta_z), 0 ],
                      [      0,                    0,         1 ]])

    rot = np.matmul(rot_x, rot_y)
    rot = np.matmul(rot, rot_z)

    return Quaternion(matrix=rot)

class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_file_path = Config.xml_path

        self.mocap = MocapDM()
        self.interface = MujocoInterface()
        self.load_mocap(Config.mocap_path)

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

        self.idx_mocap = 0
        self.reference_state_init()

        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 6)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        position = position[2:] # ignore x and y
        return np.concatenate((position, velocity))

    def reference_state_init(self):
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        self.idx_curr = 0

    def early_termination(self):
        pass

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos[3:] # to exclude root joint

    def get_joint_velocities(self):
        data = self.sim.data
        return data.qvel[3:] # to exclude root joint

    def get_root_pos(self):
        data = self.sim.data
        return data.qpos[:3]

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)

    def calc_single_config_diff(self, seg_env, seg_mocap):
        q_0 = degree2quaternion(seg_env[0], seg_env[1], seg_env[2])
        q_1 = Quaternion(seg_mocap[0], seg_mocap[1], seg_mocap[2], seg_mocap[3])

        q_diff =  q_0.conjugate * q_1
        # q_diff =  q_1 * q_0.conjugate
        axis = q_diff.axis
        angle = q_diff.angle
        
        tmp_vel = angle * axis
        vel_angular = [tmp_vel[0], tmp_vel[1], tmp_vel[2]]

        return vel_angular

    def calc_config_errs(self, env_config, mocap_config):
        curr_idx_env = 0
        curr_idx_mocap = 0
        offset_idx_env = 0 
        offset_idx_mocap = 0

        err = []

        for each_joint in BODY_JOINTS:
            curr_idx_env = offset_idx_env
            curr_idx_mocap = offset_idx_mocap

            dof = DOF_DEF[each_joint]
            if dof == 1:
                offset_idx_env += 1
                offset_idx_mocap += 1
                seg_0 = env_config[curr_idx_env:offset_idx_env]
                seg_1 = mocap_config[curr_idx_mocap:offset_idx_mocap]
                err += [(seg_1 - seg_0) * 1.0]
            elif dof == 3:
                offset_idx_env += 3
                offset_idx_mocap += 4
                seg_env = env_config[curr_idx_env:offset_idx_env]
                seg_mocap = mocap_config[curr_idx_mocap:offset_idx_mocap]
                err += self.calc_single_config_diff(seg_env=seg_env, seg_mocap=seg_mocap)
        return sum(abs(err))

    def calc_vel_errs(self, now_vel, next_vel):
        assert len(now_vel) == len(next_vel)
        err = sum(abs(np.array(now_vel) - np.array(next_vel)))
        return err

    def calc_root_errs(self, curr_root, target_root): # including root joint
        assert len(curr_root) == len(target_root)
        assert len(curr_root) == 3
        return np.sum(abs(curr_root - target_root))

    def calc_reward(self):
        assert len(self.mocap.data) != 0
        self.update_inteval = int(self.mocap_dt // self.dt)

        err_configs = 0.0
        err_vel = 0.0
        # err_end_eff = 0.0
        err_root = 0.0
        # err_com = 0.0

        if self.idx_curr % self.update_inteval != 0:
            return 0.0

        self.idx_mocap = int(self.idx_curr // self.update_inteval) + self.idx_init
        self.idx_mocap = self.idx_mocap % self.mocap_data_len

        target_config = self.mocap.data[self.idx_mocap, 1+3:] # to exclude root joint
        self.curr_frame = target_config
        curr_configs = self.get_joint_configs()

        err_configs = self.calc_config_errs(curr_configs, target_config)

        curr_mocap_vel = self.mocap.data_vel[self.idx_mocap]
        curr_vel = self.get_joint_velocities()

        err_vel = self.calc_vel_errs(curr_mocap_vel, curr_vel)

        target_root = self.mocap.data[self.idx_mocap, 1: 1+3]
        curr_root = self.get_root_pos()

        err_root = self.calc_root_errs(curr_root, target_root)

        ## TODO
        # err_end_eff =  0.0
        # reward_end_eff  = math.exp(-self.scale_err * self.scale_end_eff * err_end_eff)

        ## TODO
        # err_com = 0.0
        # reward_com      = math.exp(-self.scale_err * self.scale_com * err_com)

        reward_pose     = math.exp(-self.scale_err * self.scale_pose * err_configs)
        reward_vel      = math.exp(-self.scale_err * self.scale_vel * err_vel)
        reward_root     = math.exp(-self.scale_err * self.scale_root * err_root)

        # reward = self.weight_pose * reward_pose + self.weight_vel * reward_vel + \
        #      self.weight_end_eff * reward_end_eff + self.weight_root * reward_root + \
        #          self.weight_com * reward_com

        reward = self.weight_pose * reward_pose + self.weight_vel * reward_vel + \
            self.weight_root * reward_root

        return reward

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.idx_curr += 1
        data = self.sim.data

        observation = self._get_obs()
        reward_obs = self.calc_reward()
        reward_acs = np.square(data.ctrl).sum()

        reward = reward_obs - 0.1 * reward_acs

        info = dict()
        done = self.is_done()

        return observation, reward, done, info

    def is_done(self):
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 0.7) or (qpos[2] > 2.0))
        return done

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        self.reference_state_init()
        qpos = self.mocap.data[self.idx_init, 1:]
        if self.idx_init == self.mocap_data_len - 1: # init to last mocap frame
            root_pos_err = np.array(self.mocap.data[self.idx_init, 1:4]) - np.array(self.mocap.data[self.idx_init-1, 1:4])
            qpos_err = self.interface.calc_config_err_vec_with_root(self.mocap.data[self.idx_init-1, 4:], self.mocap.data[self.idx_init, 4:])
        else:
            root_pos_err = np.array(self.mocap.data[self.idx_init+1, 1:4]) - np.array(self.mocap.data[self.idx_init, 1:4])
            qpos_err = self.interface.calc_config_err_vec_with_root(self.mocap.data[self.idx_init, 4:], self.mocap.data[self.idx_init+1, 4:])
        qvel = np.concatenate((root_pos_err, qpos_err), axis=None) * 1.0 / self.mocap_dt
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

    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    print(action_size)
    curr_idx = 0
    while True:
        curr_idx = curr_idx % env.mocap_data_len
        target_config = env.mocap.data[curr_idx, 1+2:] # to exclude root joint
        env.sim.data.qpos[2:] = target_config[:]
        env.sim.forward()
        print(env.sim.data.qvel)
        env.render()
        curr_idx +=1