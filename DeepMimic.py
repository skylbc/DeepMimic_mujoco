import numpy as np
import sys
import random

# from env.deepmimic_env_mujoco import DeepMimicEnv
from mujoco.mujoco_env import DeepMimicEnv
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.util as Util

# anim
fps = 60
update_timestep = 1.0 / fps
animating = True

playback_speed = 1

# FPS counter
prev_time = 0
updates_per_sec = 0

args = []
world = None

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

def draw():
    world.env.draw()

def reload():
    global world
    global args

    world = build_world(args, enable_draw=True)
    return

def reset():
    world.reset()
    return

def shutdown():
    global world

    Logger.print('Shutting down...')
    world.shutdown()
    sys.exit(0)
    return

def animate():
    global world

    timestep = -update_timestep if (playback_speed < 0) else update_timestep
    update_world(world, timestep)
            
    if (world.env.is_done()):
        shutdown()

    return

def build_world(args, enable_draw, playback_speed=1):
    arg_parser = build_arg_parser(args)
    env = DeepMimicEnv()
    world = RLWorld(env, arg_parser)
    return world

def main():
    global args

    args = ["--arg_file", "args/run_humanoid3d_crawl_args.txt"]
    reload()
    while True:
        animate()
    return

if __name__ == '__main__':
    main()
