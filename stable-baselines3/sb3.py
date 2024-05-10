# CSC 570 Final Project - sb3.py

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.callbacks import BaseCallback

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import argparse
import sys
import os
from typing import Callable


class SaveBestModelCallback(BaseCallback):
    """
    Callback for saving the best performing model during training.

    :param eval_env: The environment used for evaluation.
    :param save_path: Path to save the best model weights.
    :param verbose: Whether to print debug information.
    """

    def __init__(self, eval_env, save_path, eval_freq=10000, verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_epoch_end(self) -> bool:
        """
        This method will be called by the model at the end of each epoch.

        :return: Whether or not training should continue.
        """
        if self.eval_freq > 0 and self.n_epochs % self.eval_freq == 0:
            # Evaluate the model
            mean_reward = self._evaluate_model()

            # Save the best model if the mean reward is better than before
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"Saving new best model with mean reward: {mean_reward}")

        return True

    def _evaluate_model(self) -> float:
        """
        Evaluate the current model on the evaluation environment.

        :return: The mean reward obtained by the model.
        """
        episode_rewards = []
        for _ in range(self.eval_env.num_envs):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards)


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


def make_env(world="1", level="1", render_mode="rgb_array"):
    """
    Utility function for multiprocessed env.
    """

    def _init():
        # Create environment

        env = gym_super_mario_bros.make(
            f"SuperMarioBros-{world}-{level}-v0",
            render_mode=render_mode,
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


def parse_args(input=sys.argv[1:]):
    # Parse top level flag
    removed = False
    if "-h" in input and input.index("-h") != 0:
        removed = True
        input = [x for x in input if x != "-h"]

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--inference",
        help="Run a given weights file (--weights_file) in the environment",
        action="store_true",
    )
    group.add_argument(
        "--train",
        help="Train an agent in the environment",
        action="store_true",
    )
    top_args, input = parser.parse_known_args(args=input)

    if removed:
        input.append("-h")

    # Parse options
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--world",
        help="World at which you want to set your environment",
        default="1",
        type=str,
    )
    parser.add_argument(
        "--level",
        help="Level at which you want to set your environment",
        default="1",
        type=str,
    )

    # Parse inference options
    if top_args.inference:
        parser.add_argument(
            "pretrained_weights_file",
            help="Path of weights file to be loaded and perform inference on",
            type=str,
        )
        parser.add_argument(
            "--num_envs",
            help="Number of environments to run in parallel",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--num_time_steps",
            help="Number of timesteps to run environment (negative value means run until done)",
            default=-1,
            type=int,
        )

    # Parse train options
    if top_args.train:
        parser.add_argument(
            "output_directory",
            help="Path of directory where files are to be saved",
            type=str,
        )
        parser.add_argument(
            "--pretrained_weights_file",
            help="Path of weights file to be loaded and continue training from",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--num_envs",
            help="Number of environments to run in parallel",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--num_time_steps",
            help="Number of timesteps to run environment",
            default=5e6,
            type=int,
        )
        parser.add_argument(
            "--learning_rate",
            help="Learning rate of agent",
            default=1e-4,
            type=float,
        )
        parser.add_argument(
            "--learning_rate_anneal",
            help="Enable learning rate scheduler",
            action="store_true",
        )
        parser.add_argument(
            "--save_frequency",
            help="Number of epochs before running callback",
            default=2,
            type=int,
        )

    sub_args = parser.parse_args(args=input)

    return top_args, sub_args


if __name__ == "__main__":
    # Parse arguments
    top_args, sub_args = parse_args()

    # Create vectorized environment
    render_mode = "rgb_array"
    if top_args.inference:
        render_mode = "human"
    vec_env = SubprocVecEnv(
        [
            make_env(sub_args.world, sub_args.level, render_mode)
            for _ in range(sub_args.num_envs)
        ]
    )  # Observation space is Box(0, 255, (84, 84, 1), uint8)

    # Apply wrappers to environments
    vec_env = VecMonitor(
        vec_env
    )  # Needed to retrieve ep_len_mean and ep_rew_mean datapoints that the regular Monitor wrapper usually produces
    vec_env = VecFrameStack(
        vec_env, n_stack=3
    )  # Observation space becomes Box(0, 255, (84, 84, 3), uint8)

    if top_args.inference:
        # Inference agent
        model = PPO.load(sub_args.pretrained_weights_file)

        done = [False]
        time_steps = sub_args.num_time_steps
        reward_total = 0
        observation = vec_env.reset()
        while not any(done) and (time_steps > 0 or time_steps < 0):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, _ = vec_env.step(action)
            time_steps -= 1
            reward_total += reward

            vec_env.render()

        print("time_steps:", sub_args.num_time_steps - time_steps)
        print("reward:", reward_total)

    if top_args.train:
        # Train agent
        gamma = 0.9
        gae_lambda = 1.0
        ent_coef = 0.01
        batch_size = 16
        n_epochs = 10
        n_steps = 512

        learning_rate = sub_args.learning_rate
        if sub_args.learning_rate_anneal:
            learning_rate = linear_schedule(sub_args.learning_rate)

        tensorboard_log_dir = os.path.join(sub_args.output_directory, "tensorboard_log")

        if sub_args.pretrained_weights_file:
            model = PPO.load(
                sub_args.pretrained_weights_file,
                env=vec_env,
                tensorboard_log=tensorboard_log_dir,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_steps=n_steps,
            )
        else:
            model = PPO(
                "CnnPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_steps=n_steps,
            )

        # TODO: run inference on eval_env?
        eval_env = SubprocVecEnv(
            [make_env(sub_args.world, sub_args.level, "rgb_array")]
        )
        eval_env = VecFrameStack(eval_env, n_stack=3)

        callback = SaveBestModelCallback(
            eval_env=eval_env,
            save_path=os.path.join(sub_args.output_directory, "models"),
            eval_freq=sub_args.save_frequency,
            verbose=1,
        )

        model.learn(
            total_timesteps=sub_args.num_time_steps,
            reset_num_timesteps=False,
            callback=callback,
        )

        model.save(os.path.join(sub_args.output_directory, "models", "last_model"))

    vec_env.close()
