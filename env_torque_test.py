import numpy as np
from dp_env_v3 import DPEnv

if __name__ == "__main__":
    env = DPEnv()
    env.reset_model()

    action_size = env.action_space.shape[0]
    ac = np.ones(action_size)
    
    np.set_printoptions(precision=3)

    while True:
        target_config = env.mocap.data_config[env.idx_curr][7:] # to exclude root joint
        curr_config = env.sim.data.qpos[7:]
        # print("Configs errors: ", np.sum(np.abs(target_config-curr_config)))
        ac = 0.8 * np.array(target_config - curr_config)
        # print(ac[6:9])

        # if ac[8] > 1.6:
        #     import pdb
        #     pdb.set_trace()

        # env.sim.data.qpos[7:] = target_config[:]
        # env.sim.forward()
        # print(env.calc_config_reward())
        _, rew, _, info = env.step(ac)
        # print("Rewards: ", rew)
        env.render()