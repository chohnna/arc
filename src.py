import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from   matplotlib import colors

import json
import os
from pathlib import Path

from subprocess import Popen, PIPE, STDOUT
from glob import glob

base_path='/Users/hannacho/Desktop/arc/arc-prize-2024/'
# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Reading files
training_challenges =  load_json(base_path +'arc-agi_training_challenges.json')
training_solutions =   load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges =load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path +'arc-agi_evaluation_solutions.json')

test_challenges =  load_json(base_path +'arc-agi_test_challenges.json')

print(f'Number of training challenges = {len(training_challenges)}')
print(f'Number of training solutions = {len(training_solutions)}')
print(f'Number of evaluation solutions = {len(evaluation_solutions)}')
print(f'Number of test challenges = {len(test_challenges)}')

task = training_challenges['007bbfb7']
print(task.keys())