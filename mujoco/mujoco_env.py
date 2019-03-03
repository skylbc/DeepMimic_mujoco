import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from mocap import MocapDM
from mujoco_interface import MujocoInterface

class DeepMimicEnv(object):
    def __init__(self, args, enable_draw):
        file_path = 'humanoid_deepmimic/envs/asset/humanoid_deepmimic.xml'
        with open(file_path, 'r') as fin:
            MODEL_XML = fin.read()

        self.model = load_model_from_xml(MODEL_XML)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        mocap_filepath = 'motions/humanoid3d_backflip.txt'

        self.mocap = MocapDM()
        self.mocap.load_mocap(mocap_filepath)

        self.interface = MujocoInterface()
        self.interface.init(self.sim, self.mocap.dt)

    def update(self, timestep):
        self.sim.step()

    def reset(self):
        self.sim.reset()

    def get_time(self):
        return self.sim.data.time

    def get_name(self):
        return 'test'

    # rendering and UI interface
    def draw(self):
        self.viewer.render()

    def keyboard(self, key, x, y):
        pass

    def mouse_click(self, button, state, x, y):
        pass

    def mouse_move(self, x, y):
        pass

    def reshape(self, w, h):
        pass

    def shutdown(self):
        pass

    def is_done(self):
        return False

    def set_playback_speed(self, speed):
        pass

    def set_updates_per_sec(self, updates_per_sec):
        pass

    def get_win_width(self):
        return 640

    def get_win_height(self):
        return 320

    def get_num_update_substeps(self):
        return 32

    # rl interface
    def is_rl_scene(self):
        return True

    def get_num_agents(self):
        return 1

    def need_new_action(self, agent_id):
        return True


    def record_state(self, agent_id):
        self.data = self.sim.data
        # Cartesian position of body frame
        xpos = self.data.body_xpos
        xquat = self.data.body_xquat
        cvel = self.data.cvel

        valid_xpos = self.interface.align_state(xpos)
        valid_xquat = self.interface.align_state(xquat)
        valid_cvel = self.interface.align_state(cvel)

        root_xpos = valid_xpos[0]

        total_length = 0
        total_length += self.mocap.num_joints * (self.mocap.pos_dim + self.mocap.rot_dim) + 1
        total_length += self.mocap.num_joints * (self.mocap.pos_dim + self.mocap.rot_dim - 1)

        self.state_size = total_length

        obs = np.zeros(total_length)
        obs.fill(np.nan) # fill with nan to avoid any missing data
        
        obs[0] = root_xpos[1]
        curr_idx = 1
        for i in range(self.mocap.num_joints):
            obs[curr_idx:curr_idx+3] = valid_xpos[i] - root_xpos
            curr_idx += 3
            obs[curr_idx:curr_idx+4] = valid_xquat[i]
            curr_idx += 4

        for i in range(self.mocap.num_joints):
            obs[curr_idx:curr_idx+6] = valid_cvel[i]
            curr_idx += 6
        
        return obs

    def record_goal(self, agent_id):
        return np.array([1])

    def get_action_space(self, agent_id):
        return 1
    
    def set_action(self, agent_id, action):
        return self._core.SetAction(agent_id, action.tolist())
    
    def get_state_size(self, agent_id):
        return self._core.GetStateSize(agent_id)

    def get_goal_size(self, agent_id):
        return self._core.GetGoalSize(agent_id)

    def get_action_size(self, agent_id):
        return self._core.GetActionSize(agent_id)

    def get_num_actions(self, agent_id):
        return self._core.GetNumActions(agent_id)

    def build_state_offset(self, agent_id):
        return np.array(self._core.BuildStateOffset(agent_id))

    def build_state_scale(self, agent_id):
        return np.array(self._core.BuildStateScale(agent_id))
    
    def build_goal_offset(self, agent_id):
        return np.array(self._core.BuildGoalOffset(agent_id))

    def build_goal_scale(self, agent_id):
        return np.array(self._core.BuildGoalScale(agent_id))
    
    def build_action_offset(self, agent_id):
        return np.array(self._core.BuildActionOffset(agent_id))

    def build_action_scale(self, agent_id):
        return np.array(self._core.BuildActionScale(agent_id))

    def build_action_bound_min(self, agent_id):
        return np.array(self._core.BuildActionBoundMin(agent_id))

    def build_action_bound_max(self, agent_id):
        return np.array(self._core.BuildActionBoundMax(agent_id))

    def build_state_norm_groups(self, agent_id):
        return np.array(self._core.BuildStateNormGroups(agent_id))

    def build_goal_norm_groups(self, agent_id):
        return np.array(self._core.BuildGoalNormGroups(agent_id))

    def calc_reward(self, agent_id):
        return self._core.CalcReward(agent_id)

    def get_reward_min(self, agent_id):
        return self._core.GetRewardMin(agent_id)

    def get_reward_max(self, agent_id):
        return self._core.GetRewardMax(agent_id)

    def get_reward_fail(self, agent_id):
        return self._core.GetRewardFail(agent_id)

    def get_reward_succ(self, agent_id):
        return self._core.GetRewardSucc(agent_id)

    def is_episode_end(self):
        return self._core.IsEpisodeEnd()

    def check_terminate(self, agent_id):
       return Env.Terminate(self._core.CheckTerminate(agent_id))

    def check_valid_episode(self):
        return self._core.CheckValidEpisode()

    def log_val(self, agent_id, val):
        self._core.LogVal(agent_id, float(val))
        return

    def set_sample_count(self, count):
        self._core.SetSampleCount(count)
        return

    def set_mode(self, mode):
        self._core.SetMode(mode.value)
        return