import numpy as np
from dp_env_v3 import DPEnv

if __name__ == "__main__":
    env = DPEnv()
    env.reset_model()

    action_size = env.action_space.shape[0]
    ac = np.ones(action_size)

    while True:
        target_config = env.mocap.data_config[env.idx_curr][7:] # to exclude root joint
        curr_config = env.sim.data.qpos[7:]
        # print("Configs errors: ", np.sum(np.abs(target_config-curr_config)))
        ac = 0.5 * np.array(target_config - curr_config)

        env.sim.data.qpos[7:] = target_config[:]
        env.sim.forward()
        print(env.calc_config_reward())
        # _, _, _, info = env.step(ac)
        # print("Rewards: ", info['reward_obs'])
        env.render()