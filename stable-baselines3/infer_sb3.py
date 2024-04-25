import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import argparse

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
        "SuperMarioBros-1-1-v0", render_mode="human", apply_api_compatibility=True
    )

    # Wrapper to setup action space
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)  # HACK
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))  # HACK

    # Wrapper to fix observation space (Needs a gymnasium wrapper to avoid errors)
    env.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
    )  # HACK
    env = MaxAndSkipEnv(env, 4)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, 84)
    env = gym.wrappers.FrameStack(env, 3)  # Output shape (3, 84, 84)

    # Run inference on agent
    model = PPO.load(args.weights_file)

    terminated = False
    truncated = False
    reward_total = 0
    observation, info = env.reset()
    while not terminated or truncated:
        action, _state = model.predict(
            np.array(observation)
        )  # np.array(obs) is necessary to convert LazyFrames (output type of FrameStack) before passing to model
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward_total += reward

        env.render()

    print("reward:", reward_total)
    env.close()
