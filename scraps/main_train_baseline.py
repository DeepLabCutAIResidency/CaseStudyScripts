import deeplabcut
import os, shutil
# from pathlib import Path
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset


######################################################
### Set config path
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
projectpath = os.path.dirname(config_path)
# Path.of(config_path)

modelprefix = "data_augm_baseline"

############################################################
## Set other params
NUM_SHUFFLES=1
SHUFFLE_ID=1
MAX_SNAPSHOTS=5
DISPLAY_ITERS=1000 # display loss every N iters; one iter processes one batch
SAVE_ITERS=25000 # save snapshots every n iters
MAX_ITERS=150000 #----to test only

#####################################################
# Create training dataset
# create_training_dataset(
#     config_path,
#     num_shuffles=NUM_SHUFFLES,
#     posecfg_template='/home/sofia/dlc/pose_cfg_template.yaml') # augmenter_type=None, posecfg_template=None,

##################################
# Create
try:
    shutil.copytree(
        os.path.join(projectpath, "dlc-models"),
        os.path.join(projectpath, modelprefix, "dlc-models"))
except FileExistsError:
    print("Folder exists already...")

###########################################
# Edit pose config file
train_pose_config_file,\
    test_pose_config_file, \
    snapshot_folder = deeplabcut.return_train_network_path(config_path,
                                                            shuffle=SHUFFLE_ID, 
                                                            trainingsetindex=0, # default
                                                            modelprefix=modelprefix) # default

edit_config(str(train_pose_config_file),
            {'rotation': 25,
             'rotratio': 0.4,
             'scale_jitter_lo': 0.5,
             'scale_jitter_up': 1.25,
             'mirror': False,
             'contrast':
                {'clahe': True,
                'claheratio': 0.1,
                'histeq': True,
                'histeqratio': 0.1},
             'motion_blur': True,
             'convolution':
                {'sharpen': False,
                'sharpenratio': 0.3,
                'edge': False,
                'emboss': False, #---- i think this should just be T/F
                'embossratio': 0.1},
             'grayscale': False,
             'covering': True,
             'elastic_transform': True,
             'gaussian_noise': False}) 

####################################
# Train
deeplabcut.train_network(config_path,
                            shuffle=SHUFFLE_ID,
                            max_snapshots_to_keep=MAX_SNAPSHOTS,
                            displayiters=DISPLAY_ITERS,
                            saveiters=SAVE_ITERS,
                            maxiters=MAX_ITERS,
                            gputouse=2,
                            allow_growth=True,
                            modelprefix=modelprefix) # allow_growth=True,