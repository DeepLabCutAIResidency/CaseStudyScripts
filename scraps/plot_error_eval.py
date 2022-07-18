# %%
from cgi import test
import os
import glob
import pandas as pd

config_path = "/media/data/stinkbugs-DLC-2022-07-15-SMALL/config.yaml"
dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
modelprefixes = []

for directory in dirs_in_project:
    if directory.startswith("data_augm_"):
        modelprefixes.append(directory)
modelprefixes.sort()
train_dict = {}
test_dict = {}
for modelprefix in modelprefixes:
    model_prefix = ''.join(['/media/data/stinkbugs-DLC-2022-07-15-SMALL/', modelprefix]) # modelprefix_pre = aug_
    aug_project_path = os.path.join(model_prefix, 'evaluation-results/iteration-1/') 
    #print(aug_project_path)
    csv_file = glob.glob(str(aug_project_path)+'*.csv')
    df = pd.read_csv(csv_file[0])
    train = list(df[df['Training iterations:'] == df['Training iterations:'].max()].sort_values(by=['Shuffle number'])[' Train error(px)'])
    testt = list(df[df['Training iterations:'] == df['Training iterations:'].max()].sort_values(by=['Shuffle number'])[' Test error(px)'])
    split_nesli = modelprefix.split('_')[-1]
    train_dict[split_nesli] = train
    test_dict[split_nesli] = testt
# %%
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame.from_dict(train_dict, orient='index')
df.index.rename('Model', inplace=True)

stacked = df.stack().reset_index()
stacked.rename(columns={'level_1': 'Shuffle', 0: 'Train Error Value (px)'}, inplace=True)

sns.swarmplot(data=stacked, x='Model', y='Train Error Value (px)', hue='Shuffle')
plt.legend(title = 'Shuffle',bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(rotation=45)
plt.grid()
plt.show()

# %%
df = pd.DataFrame.from_dict(test_dict, orient='index')
df.index.rename('Model', inplace=True)

stacked = df.stack().reset_index()
stacked.rename(columns={'level_1': 'Shuffle', 0: 'Test Error Value (px)'}, inplace=True)

sns.swarmplot(data=stacked, x='Model', y='Test Error Value (px)', hue='Shuffle')
plt.legend( title="Shuffle", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(rotation=45)
plt.grid()
plt.show()
# %%
