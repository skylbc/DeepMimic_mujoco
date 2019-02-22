import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

class DeepMimicEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.num_joint = np.nan
        self.pos_dim = 3
        self.rot_dim = 4

        mujoco_env.MujocoEnv.__init__(self, '/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic/envs/asset/humanoid_deepmimic.xml', 5)
        utils.EzPickle.__init__(self)

        rand_seed = np.random.randint(np.iinfo(np.int32).max)
        self.seed(rand_seed)

    def _get_joint_index(self):
        all_joint_names = ["worldbody", "root", "joint_waist", "chest", "neck", 
                           "joint_neck", "right_clavicle", "right_shoulder", "joint_right_shoulder", "right_elbow", 
                           "joint_right_elbow", "right_wrist", "left_clavicle", "left_shoulder", "joint_left_shoulder", 
                           "left_elbow", "joint_left_elbow", "left_wrist", "right_hip", "joint_right_hip", 
                           "right_knee", "joint_right_knee", "right_ankle", "joint_right_ankle", "left_hip", 
                           "joint_left_hip", "left_knee", "joint_left_knee", "left_ankle", "joint_left_ankle"]

        valid_joint_names = ["root", "chest", "neck", "right_shoulder", "right_elbow",
                             "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip",
                             "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]

        self.num_joint = len(valid_joint_names)

        idx = [a in valid_joint_names for a in all_joint_names]
        return idx

    def _update_data(self):
        self.data = self.sim.data

    def record_state(self, id):
        self._update_data()
        # Cartesian position of body frame
        xpos = self.data.body_xpos
        xquat = self.data.body_xquat
        cvel = self.data.cvel

        idx = self._get_joint_index()

        valid_xpos = xpos[idx]
        valid_xquat = xquat[idx]
        valid_cvel = cvel[idx]

        root_xpos = valid_xpos[0]

        total_length = 0
        total_length += self.num_joint * (self.pos_dim + self.rot_dim) + 1
        total_length += self.num_joint * (self.pos_dim + self.rot_dim - 1)

        self.state_size = total_length

        obs = np.zeros(total_length)
        obs.fill(np.nan) # fill with nan to avoid any missing data
        
        obs[0] = root_xpos[1]
        curr_idx = 1
        for i in range(self.num_joint):
            obs[curr_idx:curr_idx+3] = valid_xpos[i] - root_xpos
            curr_idx += 3
            obs[curr_idx:curr_idx+4] = valid_xquat[i]
            curr_idx += 4

        for i in range(self.num_joint):
            obs[curr_idx:curr_idx+6] = valid_cvel[i]
            curr_idx += 6
        
        return obs

    def calc_reward(self, agent_id):
        return 1
        '''
        pose_w = 0.5
        vel_w = 0.05
        end_eff_w = 0.15
        root_w = 0.2
        com_w = 0.1

        total_w = pose_w + vel_w + end_eff_w + root_w + com_w
        pose_w /= total_w
        vel_w /= total_w
        end_eff_w /= total_w
        root_w /= total_w
        com_w /= total_w

        pose_scale = 2
        vel_scale = 0.1
        end_eff_scale = 40
        root_scale = 5
        com_scale = 10
        err_scale = 1

        const auto& joint_mat = sim_char.GetJointMat();
        const auto& body_defs = sim_char.GetBodyDefs();
        double reward = 0;

        const Eigen::VectorXd& pose0 = sim_char.GetPose();
        const Eigen::VectorXd& vel0 = sim_char.GetVel();
        const Eigen::VectorXd& pose1 = kin_char.GetPose();
        const Eigen::VectorXd& vel1 = kin_char.GetVel();
        tMatrix origin_trans = sim_char.BuildOriginTrans();
        tMatrix kin_origin_trans = kin_char.BuildOriginTrans();

        tVector com0_world = sim_char.CalcCOM();
        tVector com_vel0_world = sim_char.CalcCOMVel();
        tVector com1_world;
        tVector com_vel1_world;
        cRBDUtil::CalcCoM(joint_mat, body_defs, pose1, vel1, com1_world, com_vel1_world);

        int root_id = sim_char.GetRootID();
        tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
        tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
        tQuaternion root_rot0 = cKinTree::GetRootRot(joint_mat, pose0);
        tQuaternion root_rot1 = cKinTree::GetRootRot(joint_mat, pose1);
        tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
        tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
        tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
        tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);

        pose_err = 0
        vel_err = 0
        end_eff_err = 0
        root_err = 0
        com_err = 0
        heading_err = 0

        num_end_effs = 0
        num_joints = sim_char.GetNumJoints();
        assert(num_joints == mJointWeights.size());

        double root_rot_w = mJointWeights[root_id];
        pose_err += root_rot_w * cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
        vel_err += root_rot_w * cKinTree::CalcRootAngVelErr(joint_mat, vel0, vel1);

        for (int j = root_id + 1; j < num_joints; ++j)
        {
            double w = mJointWeights[j];
            double curr_pose_err = cKinTree::CalcPoseErr(joint_mat, j, pose0, pose1);
            double curr_vel_err = cKinTree::CalcVelErr(joint_mat, j, vel0, vel1);
            pose_err += w * curr_pose_err;
            vel_err += w * curr_vel_err;

            bool is_end_eff = sim_char.IsEndEffector(j);
            if (is_end_eff)
            {
                tVector pos0 = sim_char.CalcJointPos(j);
                tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, j);
                double ground_h0 = mGround->SampleHeight(pos0);
                double ground_h1 = kin_char.GetOriginPos()[1];

                tVector pos_rel0 = pos0 - root_pos0;
                tVector pos_rel1 = pos1 - root_pos1;
                pos_rel0[1] = pos0[1] - ground_h0;
                pos_rel1[1] = pos1[1] - ground_h1;

                pos_rel0 = origin_trans * pos_rel0;
                pos_rel1 = kin_origin_trans * pos_rel1;

                double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
                end_eff_err += curr_end_err;
                ++num_end_effs;
            }
        }

        if (num_end_effs > 0)
        {
            end_eff_err /= num_end_effs;
        }

        double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos());
        double root_ground_h1 = kin_char.GetOriginPos()[1];
        root_pos0[1] -= root_ground_h0;
        root_pos1[1] -= root_ground_h1;
        double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
        
        double root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1);
        root_rot_err *= root_rot_err;

        double root_vel_err = (root_vel1 - root_vel0).squaredNorm();
        double root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm();

        root_err = root_pos_err
                + 0.1 * root_rot_err
                + 0.01 * root_vel_err
                + 0.001 * root_ang_vel_err;
        com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm();

        double pose_reward = exp(-err_scale * pose_scale * pose_err);
        double vel_reward = exp(-err_scale * vel_scale * vel_err);
        double end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
        double root_reward = exp(-err_scale * root_scale * root_err);
        double com_reward = exp(-err_scale * com_scale * com_err);

        reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward
            + root_w * root_reward + com_w * com_reward;

        return reward;
        '''

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = self.calc_reward(0)
        done = False
        info = dict(reward_linvel=0, 
                    reward_quadctrl=0, 
                    reward_alive=0, 
                    reward_impact=0)
        return self.record_state(0), reward, done, info

    def update(self, timestep):
        fps = 60
        update_timestep = 1.0 / fps
        self.frame_skip = int(timestep/update_timestep)
        # assert self.ac != None
        self.step(self.ac)
        self.render()

    def reset(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

    def get_time(self):
        return self.data.time

    def get_name(self):
        return 'test_mujoco'

    # rendering and UI interface
    def draw(self):
        self.render()

    def shutdown(self):
        self.close()

    def is_done(self):
        return False

    def get_num_update_substeps(self):
        return 10

    # rl interface
    def is_rl_scene(self):
        return True

    def get_num_agents(self):
        return 1

    def need_new_action(self, agent_id):
        return True

    def record_goal(self, agent_id):
        return np.array([1])
        # return np.array(self._core.RecordGoal(agent_id))

    def get_action_space(self, agent_id):
        return 1
        # return ActionSpace(self._core.GetActionSpace(agent_id))
    
    def set_action(self, agent_id, action):
        self.ac = action
    
    def get_state_size(self, agent_id):
        return self.state_size

    def get_goal_size(self, agent_id):
        return 1

    def get_action_size(self, agent_id):
        return 26

    def get_num_actions(self, agent_id):
        return 0

    def build_state_offset(self, agent_id):
        return np.zeros(self.get_state_size(agent_id))

    def build_state_scale(self, agent_id):
        return np.ones(self.get_state_size(agent_id))
    
    def build_goal_offset(self, agent_id):
        return np.zeros(1)

    def build_goal_scale(self, agent_id):
        return np.ones(1)
    
    def build_action_offset(self, agent_id):
        return np.zeros(self.get_action_size(agent_id))

    def build_action_scale(self, agent_id):
        return np.ones(self.get_action_size(agent_id))

    def build_action_bound_min(self, agent_id):
        return -10 * np.ones(self.get_action_size(agent_id))

    def build_action_bound_max(self, agent_id):
        return 10 * np.ones(self.get_action_size(agent_id))

    def build_state_norm_groups(self, agent_id):
        tmp = np.zeros(self.get_state_size(agent_id))
        tmp[-1] = 1
        return tmp

    def build_goal_norm_groups(self, agent_id):
        return np.ones(1)

    def is_episode_end(self):
        return False

    def check_terminate(self, agent_id):
        return 2

    def check_valid_episode(self):
        return True

    def log_val(self, agent_id, val):
        pass

    def set_sample_count(self, count):
        pass

    def set_mode(self, mode):
        pass

if __name__ == "__main__":
    env = DeepMimicEnv()
    env.record_state(0)
