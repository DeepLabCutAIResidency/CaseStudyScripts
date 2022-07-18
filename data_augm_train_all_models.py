"""
Script for training all shuffles of a number of models in a specific GPU
We pass as command line inptus
- config.yaml path
- string prefix identifying the subdirectories of the models we want to train, in the parent folder of the config.yaml file (e.g.: data_augm_)
- indices of the models to train from the subset that start with the input prefix, in alphabetical order
- gpu to use

Example usage:
To train the first three models in the sorted list of subdirs that start with data_augm_*, in gpu=3, run: 
    python data_augm_train_all_models.py <path_to_config.yaml> 'data_augm_' 0 2 3

Contributors: Jonas, Sofia
"""

import os
import sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config
import re 
import pdb

###########################################
def train_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        max_snapshots_to_keep=10,
                        displayiters=1000,
                        maxiters=500000,
                        saveiters=100000,
                        gputouse=0,
                        modelprefix="",
                        train_iteration=0):

    ##########################################################
    ### Get config as dict and associated paths
    cfg = read_config(config_path)
    project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
    training_datasets_path = os.path.join(project_path, "training-datasets")

    # Get list of shuffles
    iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(train_iteration))
    dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
    files_in_dataset_top_folder = os.listdir(dataset_top_folder)
    shuffle_numbers = []
    for file in files_in_dataset_top_folder:
        if file.endswith(".mat"):
            shuffleNum = int(re.findall('[0-9]+',file)[-1])
            shuffle_numbers.append(shuffleNum)
    shuffle_numbers.sort()
    
    # Train every shuffle
    for sh in shuffle_numbers:
        deeplabcut.train_network(config_path, # config.yaml, common to all models
                                shuffle=sh,
                                trainingsetindex=trainingsetindex,
                                max_snapshots_to_keep=max_snapshots_to_keep,
                                displayiters=displayiters,
                                maxiters=maxiters,
                                saveiters=saveiters,
                                gputouse=gputouse,
                                allow_growth=True,
                                modelprefix=modelprefix)


#############################################
if __name__ == "__main__":

    ## Get cli params
    config_path = str(sys.argv[1]) #"/media/data/stinkbugs-DLC-2022-07-15-SMALL/config.yaml"
    subdir_prefix_str = str(sys.argv[2]) # "data_augm_"

    # Select range to train and gpu
    first_model_index = int(sys.argv[3])
    last_model_index = int(sys.argv[4])
    gpu_to_use = int(sys.argv[5])

    ## Get other params (hardcoded for now)---maybe use argparser?
    TRAINING_SET_INDEX = 0 # default;
    MAX_SNAPSHOTS = 10
    DISPLAY_ITERS = 1000 # display loss every N iters; one iter processes one batch
    MAX_ITERS = 300000
    SAVE_ITERS = 50000 # save snapshots every n iters
    TRAIN_ITERATION = 1 # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?


    ## Compute list of subdirectories that start with 'subdir_prefix_str'
    list_all_dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
    list_models_subdird = []
    for directory in list_all_dirs_in_project:
        if directory.startswith(subdir_prefix_str):
            list_models_subdird.append(directory)
    list_models_subdird.sort() # sorts in place

    ## Train models in required indices
    for modelprefix in list_models_subdird[first_model_index:last_model_index+1]:
        train_all_shuffles(config_path, # config.yaml, common to all models
                            trainingsetindex=TRAINING_SET_INDEX,
                            max_snapshots_to_keep=MAX_SNAPSHOTS,
                            displayiters=DISPLAY_ITERS,
                            maxiters=MAX_ITERS,
                            saveiters=SAVE_ITERS,
                            gputouse=gpu_to_use,
                            modelprefix=modelprefix,
                            train_iteration=TRAIN_ITERATION)
