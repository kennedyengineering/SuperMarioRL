import gymnasium as gym

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Create environment (way 1)
env = gym.make(
    "GymV26Environment-v0",
    env_id="SuperMarioBros-1-1-v0",
    make_kwargs={"render_mode": "human", "apply_api_compatibility": True},
)

# Create environment (way 2)
# env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="human", apply_api_compatibility=True)
# env = gym.make("GymV26Environment-v0", env=env)    # Optional for this script

# Apply wrapper
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Begin environment
done = True
for step in range(5000):
    if done:
        state = env.reset()

    next_state, reward, done, trunc, info = env.step(env.action_space.sample())

    env.render()

env.close()
