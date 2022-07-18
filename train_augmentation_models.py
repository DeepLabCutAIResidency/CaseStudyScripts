"""
script for training all shuffles of a number of augmentation models
command line input = first_model, last_model, gpu_to_use

So if you run python train_augmentation_models.py 0 3 0
you would train the first four models on gpu 0
"""

import os
from train_all_shuffles import train_all_shuffles
import sys

first_model_index = int(sys.argv[1])
last_model_index = int(sys.argv[2])
gpu_to_use = int(sys.argv[3])

config_path = "/media/data/stinkbugs-DLC-2022-07-15/config.yaml"
dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
modelprefixes = []

for directory in dirs_in_project:
    if directory.startswith("data_augm_"):
        modelprefixes.append(directory)
modelprefixes.sort()

for modelprefix in modelprefixes[first_model_index:last_model_index+1]:
    train_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        max_snapshots_to_keep=10,
                        displayiters=1000,
                        maxiters=300000,
                        saveiters=50000,
                        gputouse=gpu_to_use,
                        modelprefix=modelprefix,
                        train_iteration=1)
