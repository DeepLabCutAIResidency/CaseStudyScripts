import numpy as np
from pathlib import Path
import os, sys

os.environ["DLClight"] = "True"

import deeplabcut
import shutil


# setting GPU: 
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

projectpath = "/media/data/stinkbugs-DLC-2022-07-15"
config = os.path.join(projectpath, "config.yaml")
videopath = os.path.join(projectpath, "videos")

#print("Checking labels")
#deeplabcut.check_labels(config)

print("Creating training sets")
for shuffle in range(3):
    deeplabcut.create_training_dataset(config,Shuffles=[shuffle])

    trainposeconfigfile, testposeconfigfile, snapshotfolder = deeplabcut.return_train_network_path(
            config,
            shuffle=shuffle,
        )

    #pointing to weights that all can access:
    cfg_dlc = deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
    cfg_dlc["init_weights"] = "/media/data/model-weights/resnet_v1_50.ckpt"
    deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile, cfg_dlc)
