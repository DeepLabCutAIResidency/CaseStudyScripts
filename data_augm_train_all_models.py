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
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
import re 
import pdb #-----
import argparse

###########################################
def train_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        max_snapshots_to_keep=10,
                        displayiters=1000,
                        maxiters=500000,
                        saveiters=100000,
                        gputouse=0,
                        modelprefix="",
                        train_iteration=0,
                        init_weights=""):

    ##########################################################
    ### Get config as dict and associated paths
    cfg = read_config(config_path)
    project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
    training_datasets_path = os.path.join(project_path, "training-datasets")

    ### Get list of shuffles
    iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(train_iteration))
    dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
    files_in_dataset_top_folder = os.listdir(dataset_top_folder)
    shuffle_numbers = []
    for file in files_in_dataset_top_folder:
        if file.endswith(".mat"):
            shuffleNum = int(re.findall('[0-9]+',file)[-1])
            shuffle_numbers.append(shuffleNum)
    shuffle_numbers.sort()
    
    ### Train every shuffle for this model
    for sh in shuffle_numbers:
        ## If initial weights different from default are provided: edit pose_cfg for this shuffle
        pdb.set_trace()
        if not init_weights: # if init_weights not empty
            # get path to train config for this shuffle
            one_train_pose_config_file_path,\
            _,_ = deeplabcut.return_train_network_path(config_path,
                                                       shuffle=sh,
                                                       trainingsetindex=trainingsetindex, 
                                                       modelprefix=modelprefix)
            # edit config
            edit_config(str(one_train_pose_config_file_path), 
                        {'init_weights':init_weights})

        ## Train this shuffle
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

    # ## Get command line input parameters
    # if an optional argument isn’t specified, it gets the None value (and None fails the truth test in an if statement)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", 
                        type=str,
                        help="path to config.yaml file [required]")
    parser.add_argument("subdir_prefix_str", 
                        type=str,
                        help="prefix common to all subdirectories to train [required]")
    parser.add_argument("gpu_to_use", 
                        type=int,
                        help="id of gpu to use (as given by nvidia-smi) [required]")
    parser.add_argument("-f", "--first_model_index", 
                        type=int,
                        help="index of the first model to train, in a sorted list of the subset of subdirectories. \
                        If none provided, all models in matching subdirectories are trained [optional]")
    parser.add_argument("-l", "--last_model_index", 
                        type=int,
                        help="index of the last model to train, in a sorted list of the subset of subdirectories. \
                        If none provided, all models in matching subdirectories are trained [optional]")
    parser.add_argument("-s", "--snapshot_initial_weights", 
                        type=str,
                        default='',
                        help="path to snapshot with weights to initialise the network with [optional]")
    args = parser.parse_args()
    pdb.set_trace()

    ### Extract input params: required
    config_path = args.config_path #str(sys.argv[1]) #"/media/data/stinkbugs-DLC-2022-07-15-SMALL/config.yaml"
    subdir_prefix_str = args.subdir_prefix_str #str(sys.argv[2]) # "data_augm_"
    gpu_to_use = args.gpu_to_use #int(sys.argv[5])
 
    ### Select snapshot of initial weights [optional]
    # this will be an empty string if no string passed
    initial_weights_snapshot_path = args.snapshot_initial_weights

    pdb.set_trace()

    ## Get other params (hardcoded for now)---have as optional inputs?
    TRAINING_SET_INDEX = 0 # default;
    MAX_SNAPSHOTS = 10
    DISPLAY_ITERS = 1000 # display loss every N iters; one iter processes one batch
    MAX_ITERS = 300000
    SAVE_ITERS = 50000 # save snapshots every n iters
    TRAIN_ITERATION = 1 # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?

    pdb.set_trace()

    ## Compute list of subdirectories that start with 'subdir_prefix_str'
    list_all_dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
    list_models_subdird = []
    for directory in list_all_dirs_in_project:
        if directory.startswith(subdir_prefix_str):
            list_models_subdird.append(directory)
    list_models_subdird.sort() # sorts in place

    ## Select range of subdirs to train (optional input args)
    if args.first_model_index:
        first_model_index = args.first_model_index 
    else:
        first_model_index = 0
    if args.last_model_index: 
        last_model_index = args.last_model_index 
    else:
        last_model_index = len(list_models_subdird)-1

    ## Set before training (allow growth bug)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    ## Train models in required indices
    for modelprefix in list_models_subdird[first_model_index:last_model_index+1]:

        # train all shuffles for each model
        pdb.set_trace()
        train_all_shuffles(config_path, # config.yaml, common to all models
                            trainingsetindex=TRAINING_SET_INDEX,
                            max_snapshots_to_keep=MAX_SNAPSHOTS,
                            displayiters=DISPLAY_ITERS,
                            maxiters=MAX_ITERS,
                            saveiters=SAVE_ITERS,
                            gputouse=gpu_to_use,
                            modelprefix=modelprefix,
                            train_iteration=TRAIN_ITERATION,
                            init_weights=initial_weights_snapshot_path)



    # config_path = str(sys.argv[1]) #"/media/data/stinkbugs-DLC-2022-07-15-SMALL/config.yaml"
    # subdir_prefix_str = str(sys.argv[2]) # "data_augm_"

    # # Select range to train and gpu
    # first_model_index = int(sys.argv[3])
    # last_model_index = int(sys.argv[4])
    # gpu_to_use = int(sys.argv[5])

    # # Select whether to restart training from a previous snapshot

    #--------