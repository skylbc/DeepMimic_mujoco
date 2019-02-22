import numpy as np
import sys
from env.deepmimic_env_mujoco import DeepMimicEnv
from learning.rl_world import RLWorld
from util.logger import Logger
import util.mpi_util as MPIUtil
from util.arg_parser import ArgParser
import util.util as Util

args = []
world = None

fps = 60
update_timestep = 1.0 / fps

def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ, Logger.print('Failed to load args from: ' + arg_file)

    rand_seed_key = 'rand_seed'
    if (arg_parser.has_key(rand_seed_key)):
        rand_seed = arg_parser.parse_int(rand_seed_key)
        rand_seed += 1000 * MPIUtil.get_proc_rank()
        Util.set_global_seeds(rand_seed)

    return arg_parser

def build_world(args, enable_draw, playback_speed=1):
    arg_parser = build_arg_parser(args)
    env = DeepMimicEnv()
    world = RLWorld(env, arg_parser)
    return world

def update_world(world, time_elapsed):
    num_substeps = world.env.get_num_update_substeps()
    timestep = time_elapsed / num_substeps
    num_substeps = 1 if (time_elapsed == 0) else num_substeps

    for i in range(num_substeps):
        world.update(timestep)

        valid_episode = world.env.check_valid_episode()
        if valid_episode:
            end_episode = world.env.is_episode_end()
            if (end_episode):
                world.end_episode()
                world.reset()
                break
        else:
            world.reset()
            break
    return

def run():
    global update_timestep
    global world

    done = False
    while not (done):
        update_world(world, update_timestep)

    return

def shutdown():
    global world

    Logger.print('Shutting down...')
    world.shutdown()
    return

def main():
    global args
    global world

    # Command line arguments
    args = ["--arg_file", "args/train_humanoid3d_spinkick_args.txt"]
    world = build_world(args, enable_draw=False)

    run()
    shutdown()

    return

if __name__ == '__main__':
    main()