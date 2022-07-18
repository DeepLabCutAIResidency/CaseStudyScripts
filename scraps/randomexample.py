#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:25:28 2019

@author: alex

"""

import numpy as np
from pathlib import Path
import os, sys

os.environ["DLClight"] = "True"

import deeplabcut
import shutil


# setting GPU: 
#gpuid=2
#gpuid=None
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

projectpath = "/media/data/SocialPaperDatasets/Marmoset-Mackenzie-2019-05-29"
config = os.path.join(projectpath, "config.yaml")
videopath = os.path.join(projectpath, "videos")

cfg = deeplabcut.auxiliaryfunctions.read_config(config)

"""
print("Creating training sets")
for shuffle in range(3):
    #trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.train_network_path(config,shuffle=shuffle,trainingsetindex=trainingsetindex)
    deeplabcut.create_multianimaltraining_dataset(config,Shuffles=[shuffle])
"""

pafwidth=12
for modelbase in ['goodbaseline_idchannel_fullyconnected',"goodbaseline_stride4idchannel_fullyconnected"]:

    modelsuffix=modelbase+str(pafwidth)
    if "sgd" in modelsuffix:
        Maxiter = int(2 * 10 ** 5)
        si = int(5 * 10 ** 4)
    else:
        Maxiter = int(2 * 10 ** 4)
        si = int(10 ** 4)

    dp = int(100)
    numsnapshots = int(Maxiter / si)

    modelprefix = "DLC-benchmarking" + modelsuffix
    # Copying datasets

    try:
        shutil.copytree(
            os.path.join(projectpath, "dlc-models"),
            os.path.join(projectpath, modelprefix, "dlc-models"),
        )
    except FileExistsError:
        print("Folder exists already...")

    trainingsetindex = 0
    for shuffle in [0, 1, 2]:

        trainposeconfigfile, testposeconfigfile, snapshotfolder = deeplabcut.return_train_network_path(
            config,
            shuffle=shuffle,
            modelprefix=modelprefix,
            trainingsetindex=trainingsetindex,
        )
        
        
        cfg_dlc = deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
        pts = cfg_dlc["all_joints"]  # steps3?

        PAF_graph = [[0, 1], [0, 2], [0, 3], [3, 1]]  # central bodyparts / head

        PAF_graph.extend([[2, 12], [12, 13], [2, 13], [12, 14], [13, 14]])  # body

        PAF_graph.extend(
            [[12, 4], [4, 10], [12, 6], [6, 11], [12, 10], [12, 11]]
        )  # front legs
        PAF_graph.extend([[2, 4], [2, 6]])

        PAF_graph.extend(
            [[14, 5], [5, 8], [14, 7], [7, 9], [14, 8], [14, 9]]
        )  # hind  legs

        PAF_graph.extend([[13, 5], [13, 7]])

        if "fullyconnected" in modelsuffix:
            PAF_graph = []
            for p1 in pts:
                for p2 in pts:
                    if p2[0] > p1[0]:
                        PAF_graph.append([p1[0], p2[0]])

        print(PAF_graph)
        names = cfg_dlc["all_joints_names"]
        # for p in range(14):
        #    print(p,names[p])

        for p in PAF_graph:
            print(p[0], p[1], names[p[0]], names[p[1]])

        if "stride4" in modelsuffix:
            cfg_dlc["bank3"] = 128
            cfg_dlc["bank5"] = 128
            cfg_dlc["smfactor"] = 4
            cfg_dlc["stride"] = 4

        # now set it!
        num_limbs = len(PAF_graph)
        cfg_dlc["num_limbs"] = int(num_limbs)
        cfg_dlc["partaffinityfield_graph"] = PAF_graph
        cfg_dlc["augmentationprobability"] = 0.5
        # cfg_dlc['save_iters']=5
        # cfg_dlc['display_iters']=1

        cfg_dlc["dataset_type"] = "multi-animal-imgaug"

        if "sgd" in modelsuffix:
            cfg_dlc["optimizer"] = "sgd"
            cfg_dlc["batch_size"] = 1
            cfg_dlc["fliplr"] = False
            cfg_dlc["hist_eq"] = False

            cfg_dlc["cropratio"] = 0.6
            cfg_dlc["cropfactor"] = 0.2

            cfg_dlc["rotation"] = False  # can also be an integer def. -10,10 if true.
            cfg_dlc["covering"] = False
            cfg_dlc["motion_blur"] = False  # [["k", 7],["angle", [-90, 90]]]

            cfg_dlc["elastic_transform"] = False

        else:
            cfg_dlc["optimizer"] = "adam"
            cfg_dlc["batch_size"] = 4
            cfg_dlc["fliplr"] = True
            cfg_dlc["hist_eq"] = True

            cfg_dlc["cropratio"] = 0.6
            cfg_dlc["cropfactor"] = 0.2

            cfg_dlc["rotation"] = 180  # can also be an integer def. -10,10 if true.
            cfg_dlc["covering"] = True
            cfg_dlc["motion_blur"] = [["k", 7], ["angle", [-90, 90]]]

            cfg_dlc["elastic_transform"] = True

        cfg_dlc["scmap_type"] = "plateau"  # gaussian'
        cfg_dlc["pafwidth"] = pafwidth

        cfg_dlc["save_iters"] = si
        cfg_dlc["display_iters"] = dp

        cfg_dlc["pairwise_loss_weight"] = 1.
        cfg_dlc["max_input_size"] = 1500
        cfg_dlc["scale_jitter_lo"] = 0.5
        cfg_dlc["scale_jitter_up"] = 1.2
        cfg_dlc["global_scale"] = 0.8
        cfg_dlc["partaffinityfield_predict"] = True

        if cfg_dlc["optimizer"] == "adam":
            cfg_dlc["multi_step"] = [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 500000]]
        else:
            cfg_dlc["multi_step"] = [[0.005, 10000], [0.01, 100000], [0.02, 430000]]

        cfg_dlc['init_weights']= str(snapshotfolderPRIOR)



        cfg_dlc["project_path"] = str(projectpath)
        deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile, cfg_dlc)

        cfg_dlc = deeplabcut.auxiliaryfunctions.read_plainconfig(testposeconfigfile)
        if "stride4" in modelsuffix:
            cfg_dlc["bank3"] = 128
            cfg_dlc["bank5"] = 128
            cfg_dlc["smfactor"] = 4
            cfg_dlc["stride"] = 4

        cfg_dlc["project_path"] = str(projectpath)
        cfg_dlc["num_limbs"] = int(num_limbs)
        cfg_dlc["partaffinityfield_predict"] = "True"
        cfg_dlc["dataset_type"] = "multi-animal-imgaug"
        cfg_dlc["partaffinityfield_graph"] = PAF_graph  # required for inference
        deeplabcut.auxiliaryfunctions.write_plainconfig(testposeconfigfile, cfg_dlc)

        Snapshots = np.array(
            [fn.split(".")[0] for fn in os.listdir(snapshotfolder) if "index" in fn]
        )
        try:  # check if any where found / there should be 4!!
            print("Starting training for", shuffle, trainingsetindex)
            print(snapshotfolder, Snapshots)
            assert len(Snapshots) == numsnapshots
            print("Network already trained...")
        except:
            print("Starting training for", shuffle, trainingsetindex)
            deeplabcut.train_network(
                config,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                max_snapshots_to_keep=31,
                maxiters=Maxiter,
                modelprefix=modelprefix,
                #gputouse=gpuid
            )

        cfg_dlc = deeplabcut.auxiliaryfunctions.read_plainconfig(testposeconfigfile)
        cfg_dlc["nmsradius"] = 5.0
        cfg_dlc["minconfidence"] = 0.01
        deeplabcut.auxiliaryfunctions.write_plainconfig(testposeconfigfile, cfg_dlc)

        print("Evaluating", shuffle, trainingsetindex)
        deeplabcut.evaluate_network(
                config,
                Shuffles=[shuffle],
                trainingsetindex=trainingsetindex,
                modelprefix=modelprefix,
        )

        print("Starting inference for", shuffle, trainingsetindex)
        deeplabcut.analyze_videos(config,[videopath],shuffle=shuffle,trainingsetindex=trainingsetindex,videotype='.mp4',modelprefix=modelprefix,destfolder=os.path.join(projectpath,modelprefix))
