# Super Mario Reinforcement Learning

Computational Neuroscience Final Project \
CSC 570, Spring 2024, Mugizi Robert Rwebangira

- Braedan Kennedy ([bkenne07@calpoly.edu](mailto:bkenne07@calpoly.edu))
- Luis David Garcia ([lgarc120@calpoly.edu](mailto:lgarc120@calpoly.edu))
- Briana Kuo ([brkuo@calpoly.edu](mailto:brkuo@calpoly.edu))
- Richard Rios ([rrios07@calpoly.edu](mailto:rrios07@calpoly.edu))

## Installation

```bash
./setup.sh
```

## Environment Activation

```bash
source .venv/bin/activate
```

## Go into the Stable-Baselines3 Directory

After setting up the environment, navigate to the `stable-baselines3` directory and follow the installation instructions provided in the README file there. This will ensure that all necessary dependencies and configurations are correctly set up for running the reinforcement learning agent.

## Levels Beat

The agent was trained on four levels in the Super Mario Bros NES environment. Below is the summary of the performance on each level:

| Level | Status   |
|-------|----------|
| 1-1   | Success  |
| 1-2   | Success  |
| 1-3   | Failure  |
| 1-4   | Success  |


## Demo Videos of Agent Gameplay

### Level 1-1 - Success

[![Level 1-1 - Success](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/fa0c8bd3-d0eb-4a41-b467-651c9d1fc71e)](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/fa0c8bd3-d0eb-4a41-b467-651c9d1fc71e)

**How to Train:***

```bash
python3 sb3.py --train --learning_rate 0.00001 --world 1 --level 1 --num_time_steps 4000000 logdir_w1-1_n4M_nsteps512_lr00001
```

**How to Execute with Pretrained Weights:**

```bash
python3 sb3.py --inference --world 1 --level 1 weights.zip
```

### Level 1-2 - Success

[![Level 1-2 - Success](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/5b0c314f-32a0-4e38-b5aa-d1c1733e3da4)](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/5b0c314f-32a0-4e38-b5aa-d1c1733e3da4)

**How to Train:***

```bash
python3 sb3.py --train --learning_rate 0.00001 --world 1 --level 2 --num_time_steps 4000000 --n_steps 1024 logdir_1-2_n4M_nsteps1024_lr0001
```

**How to Execute with Pretrained Weights:**

```bash
 python3 sb3.py --inference --world 1 --level 2 logdir_1-2_n4M_nsteps1024_lr0001/models/best_model.zip
```

### Level 1-3 - Failure

[![Level 1-3 - Failure](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/322a0cd5-12b7-4f10-a1c0-208eb2e7c3a7)](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/322a0cd5-12b7-4f10-a1c0-208eb2e7c3a7)

**How to Train:***

```bash
python3 sb3.py --train --learning_rate 0.00075 --world 1 --level 3 --num_time_steps 4000000 logdir_w1-3_n4M_nsteps512_lr000075
```

**How to Execute with Pretrained Weights:**

```bash
python3 sb3.py --inference --world 1 --level 3 logdir_w1-3_n4M_nsteps512_lr000075/models/best_model.zip
```

### Level 1-4 - Success

[![Level 1-4 - Success](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/18ce0a3d-9b4d-402a-89f4-cd432b8f2ea7)](https://github.com/kennedyengineering/SuperMarioRL/assets/87344382/18ce0a3d-9b4d-402a-89f4-cd432b8f2ea7)

**How to Train:***

```bash
python3 sb3.py --train --learning_rate 0.0001 --world 1 --level 4 --num_time_steps 4000000 logdir_w1-4_n4M_nsteps512_lr00001
```

**How to Execute with Pretrained Weights:**

```bash
python3 sb3.py --inference --world 1 --level 4 logdir_w1-4_n4M_nsteps512_lr00001/models/best_model.zip
```
