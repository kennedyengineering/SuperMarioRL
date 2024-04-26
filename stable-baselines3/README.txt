create virtual environment:
python3 -m venv venv
source venv/bin/activate

upgrade pip:
pip install --upgrade pip

install OpenGL:
sudo apt install mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev

install environment:
pip install gym-super-mario-bros

install StableBaselines3:
pip install stable-baselines3[extra]

[optional for GPU] if on server install correct torch package:
pip uninstall torch
pip install torch==2.2.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

train agent:
python3 sb3.py --train <output_weights.zip>

inference agent:
python3 sb3.py --inference <input_weights.zip>

more help:
python3 sb3.py -h
python3 sb3.py --train -h
python3 sb3.py --inference -h
