# Intro

Mujoco version of [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html): 
* No C++ codes --> pure python
* No bullet engine --> Mujoco engine
* No PPO --> TRPO-based 

# Usage
Training a policy:
``` bash
python3 trpo.py --task train
```
Running a policy:
``` bash
python3 trpo.py --task evaluate --load_model_path XXXX # for evaluation
```

# Acknowledge

This repository is based on code accompanying the SIGGRAPH 2018 paper:
"DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills".
The framework uses reinforcement learning to train a simulated humanoid to imitate a variety
of motion skills from mocap data.
Project page: https://xbpeng.github.io/projects/DeepMimic/index.html
