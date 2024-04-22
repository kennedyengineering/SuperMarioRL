create virtual environment:
python3 -m venv venv
source venv/bin/activate

upgrade pip:
pip install --upgrade pip

install OpenGL:
sudo apt install mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev

install environment:
pip install gym-super-mario-bros

install StableBaselines3
pip install stable-baselines3[extra]

test run evironment:
python3 test_env.py

train agent:
python3 train_sb3.py

inference agent:
python3 infer_sb3.py

NOTES:
gym_super_mario_bros.make is just an alias to gym.make for convenience.
gym-super-mario-bros==7.4.0 ships with gym==0.26.2
gym_super_mario_bros is a terminal application so you can play the game -- is hopelessly broken though...
shimmy installs gymnasium(latest) and gym==0.26.2
stable-baselines3 can take a gym==0.26.2 environment and will automatically apply compatibility wrappers (via shimmy?).
JoypadSpace creates a gym.spaces.Discrete, when stable-baselines3 checks for a gymnasium.spaces.Discrete and will throw an error.
JoypadSpace causes an issue with env.reset(), is fixable by overriding the method with a lambda.
model.predict(obs) returns action, _state. action is a numpy.ndarray, I think because stable-baselines3 trains the model using VecEnv, so an action is returned for each environment. For now, just bypassing creating a VecEnv by passing action.item() into env.step().
to use tensorboard: tensorboard --logdir <DIR>
