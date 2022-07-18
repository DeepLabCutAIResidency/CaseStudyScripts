modelprefixes = [""]

for modelprefix in modelprefixes[start:end]:
    train_all_shuffles(config_path, # config.yaml, common to all models
                        trainingsetindex=0,
                        max_snapshots_to_keep=10,
                        displayiters=1000,
                        maxiters=500000,
                        saveiters=100000,
                        gputouse=0,
                        modelprefix=modelprefix,
                        train_iteration=0)
