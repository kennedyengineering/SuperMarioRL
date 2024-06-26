import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import argparse
from typing import Callable

# TODO: add checkpointing
# TODO: framestacking with FrameStack or VecFrameStack? if FrameStack, then the make_env() method can be used in infer_sb3 and train_sb3


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def make_env():
    """
    Utility function for multiprocessed env.
    """

    def _init():
        # Create environment
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0",
            render_mode="rgb_array",
            apply_api_compatibility=True,
        )

        # Wrapper to fix action space
        JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)  # HACK
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))  # HACK

        # Wrapper to fix observation space (Needs a gymnasium wrapper to avoid errors)
        env.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
        )  # HACK
        env = MaxAndSkipEnv(env, 4)
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = gym.wrappers.ResizeObservation(env, 84)

        env.reset()

        return env

    return _init


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_file",
        help="Path of weights file to be saved",
        default="weights_vec",
        type=str,
    )
    parser.add_argument(
        "--pretrained_weights_file",
        help="Path of weights file to continue training on",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        help="Path of tensorboard log directory",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--num_envs",
        help="Number of environments to run in parallel",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--num_time_steps",
        help="Number of timesteps to run environment",
        default=1_000_000,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        help="Learning rate of agent",
        default=0.003,
        type=float,
    )
    parser.add_argument(
        "--learning_rate_anneal",
        help="Enable learning rate scheduler",
        action="store_true",
    )
    args = parser.parse_args()

    # Create vectorized environment
    vec_env = SubprocVecEnv(
        [make_env() for _ in range(args.num_envs)]
    )  # Observation space is Box(0, 255, (84, 84, 1), uint8)

    # Apply wrappers to environments
    vec_env = VecMonitor(
        vec_env
    )  # Needed to retrieve ep_len_mean and ep_rew_mean datapoints that the regular Monitor wrapper usually produces
    vec_env = VecFrameStack(
        vec_env, n_stack=3
    )  # Observation space becomes Box(0, 255, (84, 84, 3), uint8)

    # Train agent
    learning_rate = args.learning_rate
    if args.learning_rate_anneal:
        learning_rate = linear_schedule(args.learning_rate)
    if args.pretrained_weights_file:
        model = PPO.load(
            args.pretrained_weights_file,
            env=vec_env,
            tensorboard_log=args.tensorboard_log_dir,
            learning_rate=learning_rate,
        )
    else:
        model = PPO(
            "CnnPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=args.tensorboard_log_dir,
            learning_rate=learning_rate,
        )
    model.learn(total_timesteps=args.num_time_steps, reset_num_timesteps=False)
    model.save(args.weights_file)

    vec_env.close()
