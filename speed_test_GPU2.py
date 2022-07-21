"""
Script for testing the GPU allow growth fix
"""
import deeplabcut
import os
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
modelprefix = 'speed_test'
shuffle = 1

trainingsetindex=0
max_snapshots_to_keep=1
displayiters=100
maxiters=5000
saveiters=1000
gputouse=3
train_iteration=1

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

deeplabcut.train_network(config_path, # config.yaml, common to all models
                         shuffle=shuffle,
                         max_snapshots_to_keep=max_snapshots_to_keep,
                         displayiters=displayiters,
                         maxiters=maxiters,
                         saveiters=saveiters,
                         gputouse=gputouse,
                         allow_growth=True,
                         modelprefix=modelprefix)