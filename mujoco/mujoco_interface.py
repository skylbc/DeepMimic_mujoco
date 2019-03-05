import numpy as np
from pyquaternion import Quaternion

from mujoco.mocap_util import calc_angular_vel_from_quaternion
from mujoco.mocap_util import BODY_JOINTS, BODY_JOINTS_IN_DP_ORDER
from mujoco.mocap_util import DOF_DEF, PARAMS_KP_KD, PARAMS_KP_KD, BODY_DEFS

class MujocoInterface(object):
    def __init__(self):
        all_mujoco_body = ["worldbody", "root", "joint_waist", "chest", "neck", 
                             "joint_neck", "right_clavicle", "right_shoulder", "joint_right_shoulder", "right_elbow", 
                             "joint_right_elbow", "right_wrist", "left_clavicle", "left_shoulder", "joint_left_shoulder", 
                             "left_elbow", "joint_left_elbow", "left_wrist", "right_hip", "joint_right_hip", 
                             "right_knee", "joint_right_knee", "right_ankle", "joint_right_ankle", "left_hip", 
                             "joint_left_hip", "left_knee", "joint_left_knee", "left_ankle", "joint_left_ankle"]

        valid_mujoco_body = ["root", "chest", "neck", "right_shoulder", "right_elbow",
                               "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip",
                               "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]

        self.idx_valid_joint = np.array([a in valid_mujoco_body for a in all_mujoco_body])

        assert len(BODY_DEFS) == len(valid_mujoco_body)

        perm_idx = []
        for each_body in BODY_DEFS:
            tmp_idx = valid_mujoco_body.index(each_body)
            perm_idx.append(tmp_idx)
        self.idx_align_perm = perm_idx

        dofs = np.fromiter(DOF_DEF.values(), dtype=np.int16)
        self.action_size = sum(dofs) + sum(dofs == 3)

        self.offset_map_dp2mujoco_pos = {}
        self.offset_map_dp2mujoco_vel = {}
        offset_idx_pos = 0
        offset_idx_vel = 0
        for each_joint in BODY_JOINTS_IN_DP_ORDER:
            self.offset_map_dp2mujoco_pos[each_joint] = offset_idx_pos
            self.offset_map_dp2mujoco_vel[each_joint] = offset_idx_vel
            if DOF_DEF[each_joint] == 1:
                offset_idx_pos += 1
                offset_idx_vel += 1
            elif DOF_DEF[each_joint] == 3:
                offset_idx_pos += 4
                offset_idx_vel += 3
            else:
                raise NotImplementedError

        self.offset_map_mujoco2dp_pos = {}
        self.offset_map_mujoco2dp_vel = {}
        offset_idx_pos = 0
        offset_idx_vel = 0
        for each_joint in BODY_JOINTS:
            self.offset_map_mujoco2dp_pos[each_joint] = offset_idx_pos
            self.offset_map_mujoco2dp_vel[each_joint] = offset_idx_vel
            if DOF_DEF[each_joint] == 1:
                offset_idx_pos += 1
                offset_idx_vel += 1
            elif DOF_DEF[each_joint] == 3:
                offset_idx_pos += 4
                offset_idx_vel += 3
            else:
                raise NotImplementedError

        kp, kd = [], []
        for each_joint in BODY_JOINTS:
            kp += [PARAMS_KP_KD[each_joint][0] for _ in range(DOF_DEF[each_joint])]
            kd += [PARAMS_KP_KD[each_joint][1] for _ in range(DOF_DEF[each_joint])]

        self.kp = np.array(kp)
        self.kd = np.array(kd)

    def init(self, sim, dt):
        self.sim = sim
        self.dt = dt

    def get_curr_pos_vel(self):
        pos = self.sim.data.qpos[7:] # supposed to be 36D
        vel = self.sim.data.qvel[6:] # supposed to be 28D

        return pos, vel

    def expant_raw_pos(self, pos_input):
        pos_output = []
        pos_output += pos_input[:7].tolist()
        curr_offset = 7
        for each_joint in BODY_JOINTS:
            if DOF_DEF[each_joint] == 1:
                pos_output += [pos_input[curr_offset]]
                curr_offset += 1
            elif DOF_DEF[each_joint] == 3:
                pos_output += xyzrot2quat(pos_input[curr_offset:curr_offset+3])
                curr_offset += 3
        return np.array(pos_output)

    def action2torque(self, action): # PD controller
        action = self.align(action, mode='dp2mujoco', opt='pos')

        curr_pos, curr_vel = self.get_curr_pos_vel()
        assert len(curr_pos) == len(action)

        p_err = self.calc_pos_err(curr_pos, action)
        vel = p_err * 1.0 / self.dt
        v_err = self.calc_vel_err(curr_vel, vel)
        torque = self.kp * p_err + self.kd * v_err
        return torque

    def align_state(self, input_val):
        valid_input_val = np.array(input_val)[self.idx_valid_joint]
        return valid_input_val[self.idx_align_perm]

    def align_ob_pos(self, ob_pos):
        return self.align(ob_pos, mode='mujoco2dp', opt='pos')

    def align_ob_vel(self, ob_vel):
        return self.align(ob_vel, mode='mujoco2dp', opt='vel')

    def calc_pos_err(self, now_pos, next_pos):
        curr_idx = 0
        offset_idx = 0
        assert len(now_pos) == len(next_pos)
        err = []

        for each_joint in BODY_JOINTS:
            curr_idx = offset_idx
            dof = DOF_DEF[each_joint]
            if dof == 1:
                offset_idx += 1
                seg_0 = now_pos[curr_idx]
                seg_1 = next_pos[curr_idx]
                err += [(seg_1 - seg_0) * 1.0]
            elif dof == 3:
                offset_idx += 4
                seg_0 = now_pos[curr_idx:offset_idx]
                seg_1 = next_pos[curr_idx:offset_idx]
                err += calc_angular_vel_from_quaternion(seg_0, seg_1, 1.0)
        return np.array(err)

    def align(self, input_val, mode, opt):
        assert opt in ['vel', 'pos']
        assert mode in ['dp2mujoco', 'mujoco2dp']

        if mode == 'dp2mujoco':
            this_joints = BODY_JOINTS
            if opt == 'vel':
                this_map = self.offset_map_dp2mujoco_vel
                this_offset = 3
            elif opt == 'pos':
                this_map = self.offset_map_dp2mujoco_pos
                this_offset = 4
            else:
                raise NotImplementedError
        elif mode == 'mujoco2dp':
            this_joints = BODY_JOINTS_IN_DP_ORDER
            if opt == 'vel':
                this_map = self.offset_map_mujoco2dp_vel
                this_offset = 3
            elif opt == 'pos':
                this_map = self.offset_map_mujoco2dp_pos
                this_offset = 4
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        output_val = []
        for each_joint in this_joints:
            offset_idx = this_map[each_joint]
            dof = DOF_DEF[each_joint]
            tmp_seg = []
            if dof == 1:
                tmp_seg = [input_val[offset_idx]]
            elif dof == 3:
                tmp_seg = input_val[offset_idx:offset_idx+this_offset].tolist()
            else:
                raise NotImplementedError
            output_val += tmp_seg

        return output_val

    def calc_vel_err(self, now_vel, next_vel):
        return next_vel - now_vel