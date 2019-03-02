import numpy as np
from pyquaternion import Quaternion

class PostProcessing(object):
    BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
                "left_shoulder", "left_elbow", "right_hip", "right_knee", 
                "right_ankle", "left_hip", "left_knee", "left_ankle"]

    BODY_JOINTS_IN_DP_ORDER = ["chest", "neck", "right_hip", "right_knee",
                            "right_ankle", "right_shoulder", "right_elbow", "left_hip", 
                            "left_knee", "left_ankle", "left_shoulder", "left_elbow"]

    DOF_DEF = {"chest": 3, "neck": 3, "right_shoulder": 3, "right_elbow": 1, 
            "left_shoulder": 3, "left_elbow": 1, "right_hip": 3, "right_knee": 1, 
            "right_ankle": 3, "left_hip": 3, "left_knee": 1, "left_ankle": 3}

    PARAMS_KP_KD = {"chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30], 
            "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50], 
            "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}

    def __init__(self):
        self.offset_map = {}
        offset_idx = 0
        for each_joint in self.BODY_JOINTS_IN_DP_ORDER:
            self.offset_map[each_joint] = offset_idx
            if self.DOF_DEF[each_joint] == 1:
                offset_idx += 1
            elif self.DOF_DEF[each_joint] == 3:
                offset_idx += 4
            else:
                raise NotImplementedError

        kp, kd = [], []
        for each_joint in self.BODY_JOINTS:
            kp += [self.PARAMS_KP_KD[each_joint][0] for _ in range(self.DOF_DEF[each_joint])]
            kd += [self.PARAMS_KP_KD[each_joint][1] for _ in range(self.DOF_DEF[each_joint])]

        self.kp = np.array(kp)
        self.kd = np.array(kd)

    def action2torque(self, action):
        pos_err = calc_pos_err_test()
        vel_err = calc_vel_err_test()
        while True:
            for (p_err, v_err) in zip(pos_err, vel_err):
                torque = kp * p_err[6:] + kd * v_err[6:]
                # torque = kp * p_err[6:]
                print(torque)
                sim.data.ctrl[:] = torque[:]
                # sim.forward()
                sim.step()
                viewer.render()
        pass

    def align_rotation(self, rot):
        q_input = Quaternion(rot[0], rot[1], rot[2], rot[3])
        q_align_right = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                                    [0.0, 0.0, 1.0], 
                                                    [0.0, -1.0, 0.0]]))
        q_align_left = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                                [0.0, 0.0, -1.0], 
                                                [0.0, 1.0, 0.0]]))
        q_output = q_align_left * q_input * q_align_right
        return q_output.elements

    def align_position(self, pos):
        assert len(pos) == 3
        left_matrix = np.array([[1.0, 0.0, 0.0], 
                                [0.0, 0.0, -1.0], 
                                [0.0, 1.0, 0.0]])
        pos_output = np.matmul(left_matrix, pos)
        return pos_output

    def calc_angular_vel_from_frames(self, orien_0, orien_1, dt):
        seg0 = self.align_rotation(orien_0)
        seg1 = self.align_rotation(orien_1)

        q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
        q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

        q_diff =  q_0.conjugate * q_1
        # q_diff =  q_1 * q_0.conjugate
        axis = q_diff.axis
        angle = q_diff.angle
        
        tmp_vel = (angle * 1.0)/dt * axis
        vel_angular = np.array([tmp_vel[0], tmp_vel[1], tmp_vel[2]])

        return vel_angular

    def calc_pos_err(self, now_pos, next_pos):
        curr_idx = 0
        offset_idx = 6
        assert len(now_pos) == len(next_pos)
        err = np.zeros_like(now_pos)

        for each_joint in self.BODY_JOINTS:
            curr_idx = offset_idx
            dof = self.DOF_DEF[each_joint]
            if dof == 1:
                offset_idx += 1
                seg_0 = now_pos[curr_idx:offset_idx]
                seg_1 = next_pos[curr_idx:offset_idx]
                err[curr_idx:offset_idx] = (seg_1 - seg_0) * 1.0
            elif dof == 3:
                offset_idx += 4
                seg_0 = now_pos[curr_idx:offset_idx]
                seg_1 = next_pos[curr_idx:offset_idx]
                err[curr_idx:offset_idx] = self.calc_angular_vel_from_frames(seg_0, seg_1, 1.0)
        return err

    def align_pos(self, pos_input):
        pos_output = []
        for each_joint in self.BODY_JOINTS:
            offset_idx = self.offset_map[each_joint]
            dof = self.DOF_DEF[each_joint]
            tmp_seg = []
            if dof == 1:
                tmp_seg = [pos_input[offset_idx]]
            elif dof == 3:
                tmp_seg = pos_input[offset_idx:offset_idx+dof+1]
            else:
                raise NotImplementedError
            pos_output += tmp_seg

        return pos_output