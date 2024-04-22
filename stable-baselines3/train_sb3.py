import gymnasium as gym

from stable_baselines3 import PPO

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import cv2
import argparse

# TODO: add checkpointing
# TODO: merge train_sb3_vec, train_sb3_ and infer_sb3 into a single script


class ShowObservation(gym.ObservationWrapper):
    def __init__(self, env, win_name="Observation"):
        super().__init__(env)
        self.win_name = win_name

    def observation(self, observation):
        cv2.imshow(self.win_name, cv2.cvtColor(observation, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        return observation


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_file",
        help="Path of weights file to be saved",
        default="weights",
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
        "--num_time_steps",
        help="Number of timesteps to run environment",
        default=1_000_000,
        type=int,
    )
    args = parser.parse_args()

    # Create environment
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True
    )

    # Wrapper to setup action space
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)  # HACK
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))  # HACK

    # Wrapper to fix observation space (Needs a gymnasium wrapper to avoid errors)
    env.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
    )  # HACK
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    # env = ShowObservation(env, "gray")
    env = gym.wrappers.ResizeObservation(env, 84)
    # env = ShowObservation(env, "scale")
    env = gym.wrappers.FrameStack(env, 3)

    # Train agent
    if args.pretrained_weights_file:
        model = PPO.load(
            args.pretrained_weights_file,
            env=env,
            tensorboard_log=args.tensorboard_log_dir,
        )
    else:
        model = PPO(
            "CnnPolicy", env=env, verbose=1, tensorboard_log=args.tensorboard_log_dir
        )
    model.learn(total_timesteps=args.num_time_steps, reset_num_timesteps=False)
    model.save(args.weights_file)

    env.close()
