#!/usr/bin/env python3

import os
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import json
import copy
import numpy as np
from pyquaternion import Quaternion

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
               "left_shoulder", "left_elbow", "right_hip", "right_knee", 
               "right_ankle", "left_hip", "left_knee", "left_ankle"]

BODY_JOINTS_IN_DP_ORDER = ["chest", "neck", "right_hip", "right_knee",
                           "right_ankle", "right_shoulder", "right_elbow", "left_hip", 
                           "left_knee", "left_ankle", "left_shoulder", "left_elbow"]

DOF_DEF = {"chest": 3, "neck": 3, "right_shoulder": 3, "right_elbow": 1, 
           "left_shoulder": 3, "left_elbow": 1, "right_hip": 3, "right_knee": 1, 
           "right_ankle": 3, "left_hip": 3, "left_knee": 1, "left_ankle": 3}

file_path = 'humanoid_deepmimic/envs/asset/humanoid_deepmimic.xml'
with open(file_path) as fin:
    MODEL_XML = fin.read()

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)


def calc_vel_from_frames(frame_0, frame_1, dt):
    curr_idx = 0
    offset_idx = 7 # root joint offset: 3 (position) + 4 (orientation)
    vel = []
    for each_joint in BODY_JOINTS:
        curr_idx = offset_idx
        dof = DOF_DEF[each_joint]
        if dof == 1:
            offset_idx += dof
            tmp_vel = (frame_1[curr_idx:offset_idx] - frame_0[curr_idx:offset_idx])*1.0/dt
            vel += [tmp_vel[0]]
        elif dof == 3:
            offset_idx = offset_idx + dof + 1
            seg0 = frame_0[curr_idx:offset_idx]
            seg0 = align_rotation(seg0)

            seg1 = frame_1[curr_idx:offset_idx]
            seg1 = align_rotation(seg1)

            q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
            q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

            q_diff = q_0.conjugate * q_1
            axis = q_diff.axis
            angle = q_diff.angle
            
            tmp_vel = (angle * 1.0)/dt * axis
            vel += [tmp_vel[0], tmp_vel[1], tmp_vel[2]]

    return np.array(vel)

def calc_angular_vel_from_frames(orien_0, orien_1, dt):
    seg0 = align_rotation(orien_0)
    seg1 = align_rotation(orien_1)

    q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
    q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

    q_diff =  q_1 * q_0.conjugate
    axis = q_diff.axis
    angle = q_diff.angle
    
    tmp_vel = (angle * 1.0)/dt * axis
    vel_angular = np.array([tmp_vel[0], tmp_vel[1], tmp_vel[2]])

    return vel_angular

def calc_linear_vel_from_frames(frame_0, frame_1, dt):
    curr_idx = 0
    offset_idx = 0 # root joint offset: 3 (position) + 4 (orientation)
    vel_linear = []

    curr_idx = offset_idx
    offset_idx += 3 # position is 3D
    vel_linear = (frame_1[curr_idx:offset_idx] - frame_0[curr_idx:offset_idx])*1.0/dt
    vel_linear = align_position(vel_linear)

    return vel_linear


def align_rotation(rot):
    q_input = Quaternion(rot[0], rot[1], rot[2], rot[3])
    q_align_right = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                                [0.0, 0.0, 1.0], 
                                                [0.0, -1.0, 0.0]]))
    q_align_left = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                               [0.0, 0.0, -1.0], 
                                               [0.0, 1.0, 0.0]]))
    q_output = q_align_left * q_input * q_align_right
    return q_output.elements

def align_position(pos):
    assert len(pos) == 3
    left_matrix = np.array([[1.0, 0.0, 0.0], 
                            [0.0, 0.0, -1.0], 
                            [0.0, 1.0, 0.0]])
    pos_output = np.matmul(left_matrix, pos)
    return pos_output

def read_positions():
    motions = None
    all_states = []

    durations = []

    with open('./motions/humanoid3d_walk.txt') as fin:
        data = json.load(fin)
        motions = np.array(data["Frames"])
        total_time = 0.0
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

    return all_states, durations

def read_velocities():
    states, durations = read_positions()

    offset_root_joint_vel = 6 # 6 = 3 (linear velocity) + 3 (angular velocity)
    n_states = len(states)
    n_vel = sum(DOF_DEF.values()) + offset_root_joint_vel 
    velocities = np.zeros((n_states, n_vel))

    for idx in range(n_states - 1):
        state_0 = states[idx]
        state_1 = states[idx+1]
        dt = durations[idx]

        curr_idx = 0
        offset_idx = 6
        
        velocities[idx, :3] = calc_linear_vel_from_frames(state_0["root_pos"], state_1["root_pos"], dt)
        velocities[idx, 3:offset_idx] = calc_angular_vel_from_frames(state_0["root_rot"], state_1["root_rot"], dt)

        for each_joint in BODY_JOINTS:
            curr_idx = offset_idx
            pos_0 = state_0[each_joint]
            pos_1 = state_1[each_joint]
            dof = DOF_DEF[each_joint]

            if dof == 1:
                offset_idx += 1
                velocities[idx, curr_idx:offset_idx] = (pos_1 - pos_0) * 1.0 / dt
            elif dof == 3:
                offset_idx += 3
                velocities[idx, curr_idx:offset_idx] = calc_angular_vel_from_frames(pos_0, pos_1, dt)

    return velocities

def render_from_pos():
    states, durations = read_positions()

    from time import sleep

    while True:
        for k in range(len(states)):
            state = states[k]
            dura = durations[k]
            sim_state = sim.get_state()

            sim_state.qpos[:3] = state['root_pos']
            sim_state.qpos[3:7] = state['root_rot']

            for each_joint in BODY_JOINTS:
                idx = sim.model.get_joint_qpos_addr(each_joint)
                tmp_val = state[each_joint]
                if isinstance(idx, np.int32):
                    assert 1 == len(tmp_val)
                    sim_state.qpos[idx] = state[each_joint]
                elif isinstance(idx, tuple):
                    assert idx[1] - idx[0] == len(tmp_val)
                    sim_state.qpos[idx[0]:idx[1]] = state[each_joint]

            # print(sim_state.qpos)
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()

            # sleep(dura)

        if os.getenv('TESTING') is not None:
            break

def render_from_vel():
    velocities = read_velocities()

    while True:
        for each_vel in velocities:
            sim_state = sim.get_state()
            sim_state.qvel[:] = each_vel[:]
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()

        if os.getenv('TESTING') is not None:
            break
    pass

if __name__ == "__main__":
    # render_from_pos()
    render_from_vel()