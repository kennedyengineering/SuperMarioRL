import gymnasium as gym

from stable_baselines3 import PPO

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import cv2
import argparse


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(
            observation, dsize=self.shape, interpolation=cv2.INTER_CUBIC
        )

        return observation


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights_file",
        help="Path of weights file to be loaded",
        type=str,
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
    env = gym.wrappers.ResizeObservation(env, 84)
    env = gym.wrappers.FrameStack(env, 3)

    # Run inference on agent
    model = PPO.load(args.weights_file)

    terminated = False
    truncated = False
    observation, info = env.reset()
    while not terminated or truncated:
        action, _state = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action.item())

        env.render()

    env.close()
