import os

import pandas as pd
from torch import nn


def init_cnn(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
