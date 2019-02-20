import mujoco_py
import numpy as np


if __name__ == "__main__":
    file_path = "/home/mingfei/Documents/DeepMimic/mujoco/humanoid_deepmimic.xml"
    model = mujoco_py.load_model_from_path(file_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    while(1):
        sim.step()
        viewer.render()
