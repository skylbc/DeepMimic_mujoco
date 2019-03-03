# Intro

Mujoco version of [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html); no C++, no bullet engine. 

# Running
**Under progress**
``` python
python3 DeepMimic.py # for evaluation
python3 DeepMimic_Optimizer.py # for training
python3 -m mujoco.mocap # for playing mocap data
```

# Acknowledge

This repository is based on code accompanying the SIGGRAPH 2018 paper:
"DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills".
The framework uses reinforcement learning to train a simulated humanoid to imitate a variety
of motion skills from mocap data.

Project page: https://xbpeng.github.io/projects/DeepMimic/index.html

![Skills](images/teaser.png)
