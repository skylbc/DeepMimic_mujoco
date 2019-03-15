#!/usr/bin/env python3
import os
import json
import copy
import numpy as np
from os import getcwd
from pyquaternion import Quaternion
from mujoco.mocap_util import align_position, align_rotation
from mujoco.mocap_util import BODY_JOINTS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_DEFS

class MocapDM(object):
    def __init__(self):
        self.num_bodies = len(BODY_DEFS)
        self.pos_dim = 3
        self.rot_dim = 4

    def load_mocap(self, filepath):
        self.read_raw_data(filepath)
        self.convert_raw_data()

    def read_raw_data(self, filepath):
        motions = None
        all_states = []

        durations = []

        with open(filepath, 'r') as fin:
            data = json.load(fin)
            motions = np.array(data["Frames"])
            m_shape = np.shape(motions)
            self.data = np.full(m_shape, np.nan)

            total_time = 0.0
            self.dt = motions[0][0]
            for each_frame in motions:
                duration = each_frame[0]
                each_frame[0] = total_time
                total_time += duration
                durations.append(duration)

            for each_frame in motions:
                curr_idx = 1
                offset_idx = 8
                state = {}
                state['root_pos'] = align_position(each_frame[curr_idx:curr_idx+3])
                state['root_rot'] = align_rotation(each_frame[curr_idx+3:offset_idx])
                for each_joint in BODY_JOINTS_IN_DP_ORDER:
                    curr_idx = offset_idx
                    dof = DOF_DEF[each_joint]
                    if dof == 1:
                        offset_idx += 1
                        state[each_joint] = each_frame[curr_idx:offset_idx]
                    elif dof == 3:
                        offset_idx += 4
                        state[each_joint] = align_rotation(each_frame[curr_idx:offset_idx])
                all_states.append(state)

        self.all_states = all_states
        self.durations = durations

    def convert_raw_data(self):
        for k in range(len(self.all_states)):
            state = self.all_states[k]
            dura = self.durations[k]

            # time duration
            init_idx = 0
            offset_idx = 1
            self.data[k, init_idx:offset_idx] = dura

            # root pos
            init_idx = offset_idx
            offset_idx += 3
            self.data[k, init_idx:offset_idx] = np.array(state['root_pos'])

            # root rot
            init_idx = offset_idx
            offset_idx += 4
            self.data[k, init_idx:offset_idx] = np.array(state['root_rot'])

            for each_joint in BODY_JOINTS:
                init_idx = offset_idx
                tmp_val = state[each_joint]
                if DOF_DEF[each_joint] == 1:
                    assert 1 == len(tmp_val)
                    offset_idx += 1
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                elif DOF_DEF[each_joint] == 3:
                    assert 4 == len(tmp_val)
                    offset_idx += 4
                    self.data[k, init_idx:offset_idx] = state[each_joint]

    def play(self, mocap_filepath):
        from mujoco_py import load_model_from_xml, MjSim, MjViewer

        xmlpath = '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/dp_env_v1.xml'
        with open(xmlpath) as fin:
            MODEL_XML = fin.read()

        model = load_model_from_xml(MODEL_XML)
        sim = MjSim(model)
        viewer = MjViewer(sim)

        self.read_raw_data(mocap_filepath)
        self.convert_raw_data()

        from time import sleep

        phase_offset = np.array([0.0, 0.0, 0.0])

        while True:
            for k in range(len(self.data)):
                tmp_val = self.data[k, 1:]
                sim_state = sim.get_state()
                sim_state.qpos[:] = tmp_val[:]
                sim_state.qpos[:3] +=  phase_offset[:]
                sim.set_state(sim_state)
                sim.forward()
                viewer.render()

            sim_state = sim.get_state()
            phase_offset = sim_state.qpos[:3]
            phase_offset[2] = 0

if __name__ == "__main__":
    test = MocapDM()
    curr_path = getcwd()
    test.play(curr_path + "/mujoco/motions/humanoid3d_crawl.txt")