# In the future we might need to define exact train and test sets to compare different models. This script has initiated for this purpose.

import os
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images
root_folder = '/home/sabrina/stinkbugs/labeled-data/'

directory_list = list()
for root, dirs, files in os.walk("/home/sabrina/stinkbugs/labeled-data", topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))

folders = [os.path.join(root_folder, x) for x in directory_list]
all_images = [img for folder in folders for img in load_images_from_folder(folder)]
# %%
n_frames = len(all_images)
frames = list(range(0,n_frames))
random.Random(4).shuffle(frames)
split =  95
percent = int(split * len(frames) / 100)
train1 = frames[:percent]
test1 = frames[percent:]

list1 = deeplabcut.create_training_dataset(
    config_path,
    num_shuffles= suffle,
    net_type="resnet_50",
    augmenter_type="imgaug",
    trainIndices = [train1],
    testIndices = [test1],
)
