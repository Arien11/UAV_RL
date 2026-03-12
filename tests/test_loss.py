import sys
import os
from tkinter.constants import FALSE
import numpy as np
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algos.loss.loss_function import LossFunction

def test_loss():
    loss_func = LossFunction()
    test_state = torch.randn(1, 3, 3)
    test_prediction = torch.randn(1, 3, 3)
    test_goal = torch.randn(1, 3)
    test_map_id = torch.zeros(0, dtype=torch.int32).unsqueeze(0)
    smoothness_cost, safety_cost, goal_cost, acceleration_cost = loss_func(test_state, test_prediction, test_goal, test_map_id)

    print(smoothness_cost, safety_cost, goal_cost, acceleration_cost)
    

if __name__ == '__main__':
    test_loss()