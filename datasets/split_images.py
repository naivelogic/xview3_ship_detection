import os
from pathlib import Path
import json

import warnings
warnings.filterwarnings("ignore")


import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd

import sys
sys.path.append("../repos/xview3-reference/reference/")
from dataloader import XView3Dataset

image_folder = '/mnt/mnt_xview3/datasets/'
label_file_root = '/mnt/mnt_xview3/datasets/labels'
chips_path = '/mnt/mnt_xview3train/datasets/train_chip200_sets/set_08'
overwrite_preproc = True #False
channels = ['vh']

# Paths defined in accordance with instructions above; should not need to change 
data_root = Path(image_folder) / 'train'
label_file = Path(label_file_root) / 'train.csv'
chips_path = Path(chips_path) #/ 'validation'

print("===================================")
print(">> Preprocessing Train Chips")
print("chip set path: ", chips_path)
print(".....")

scene_ids = os.listdir(data_root)[188:300] #['356a255bd8e6ba21t']
print("Lenth of dataset IDs: ", len(scene_ids))

xview_data = XView3Dataset(
        data_root,
        None,
        "train",
        chips_path=chips_path,
        detect_file=label_file,
        scene_list=scene_ids,
        background_frac=0.0,
        overwrite_preproc=overwrite_preproc,
        channels=channels,
        chip_size=200,
        num_workers=1
    )

print("===================================")
print(">> Preprocessing Complete for Train Chips")
print("chip set path: ", chips_path)
print("")
print("scene list: ", scene_ids)

