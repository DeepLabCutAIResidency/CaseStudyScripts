import os
from train_all_shuffles import train_all_shuffles

config_path = "/media/data/stinkbugs-DLC-2022-07-15/config.yaml"
dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
modelprefixes = []

for directory in dirs_in_project:
    if directory.startswith("data_augm_"):
        modelprefixes.append(directory)
modelprefixes.sort()

for modelprefix in modelprefixes:
    train_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        max_snapshots_to_keep=10,
                        displayiters=1,
                        maxiters=10,
                        saveiters=5,
                        gputouse=0,
                        modelprefix=modelprefix,
                        train_iteration=1)
