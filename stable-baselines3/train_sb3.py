import gymnasium as gym

from stable_baselines3 import PPO

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import cv2


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


# Create environment
env = gym_super_mario_bros.make(
    "SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True
)

# Apply Wrappers to environment
# Wrapper to setup action space
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)  # HACK
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))  # HACK
# Wrapper to fix observation space
env = ResizeObservation(env, shape=84)

# Train agent
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("weights")

env.close()
