import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import argparse


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
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = gym.wrappers.ResizeObservation(env, 84)

        env.reset()

        return env

    return _init


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_file",
        help="Path of weights file to be saved",
        default="weights_vec",
        type=str,
    )
    parser.add_argument(
        "--num_envs",
        help="Number of environments to run in parallel",
        default=5,
        type=int,
    )
    args = parser.parse_args()

    # Create vectorized environment
    vec_env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])

    # Apply wrappers to environments
    vec_env = VecFrameStack(vec_env, n_stack=3)

    # Train agent
    model = PPO("CnnPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save(args.save_file)

    vec_env.close()
