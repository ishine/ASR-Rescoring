import numpy as np
import torch
from util.minimum_edit_distance import minimum_edit_distance

a = np.array([100, 110, 150])
print(np.exp(a) / np.sum(np.exp(a)))
b = a - np.average(a)
b = np.exp(b)
print(b / np.sum(b))