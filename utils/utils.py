import random
from base64 import b64encode

import numpy as np
import torch


def set_seed(env, seed):
    """Helper function to set the seeds when needed"""
    env.seed(seed)  # Environment seed
    env.action_space.seed(seed)  # Seed for env.action_space.sample()
    np.random.seed(seed)  # Numpy seed
    torch.manual_seed(seed)  # PyTorch seed
    random.seed(seed)  # seed for Python random library


class ColabVideo():
    def __init__(self, path):
        # Source: https://stackoverflow.com/a/57378660/3890306
        self.video_src = 'data:video/mp4;base64,' + b64encode(open(path, 'rb').read()).decode()

    def _repr_html_(self):
        return """
        <video width=400 controls>
              <source src="{}" type="video/mp4">
        </video>
        """.format(self.video_src)
