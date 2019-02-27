import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

class Humanoid3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.num_joint = np.nan
        self.pos_dim = 3
        self.rot_dim = 4

        self.counter = 0
        self.frame_skip = 1

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

    def _get_obs(self):
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

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = (np.random.rand() - 0.5) * 2
        done = False
        info = dict(reward_linvel=0, 
                    reward_quadctrl=0, 
                    reward_alive=0, 
                    reward_impact=0)
        return self._get_obs(), reward, done, info

    def update(self, timestep):
        self.frame_skip = 5
        act = 4 * (np.random.rand(self.get_action_size()) - 0.5)
        self.step(act)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._ezpickle_args

    def get_time(self):
        return self.data.time
    
    def set_action(self, action):
        self.ac = action
        self.step(self.ac)
    
    def get_state_size(self):
        return self.state_size

    def get_goal_size(self):
        return 1

    def get_action_size(self):
        return 26

    def get_num_actions(self):
        return 0

if __name__ == "__main__":
    env = Humanoid3DEnv()
    while True:
        fps = 60
        update_timestep = 5.0 / fps
        env.update(update_timestep)
        env.render()
