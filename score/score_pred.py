
import pandas as pd
import json
import numpy as np
from pandas.io.json import json_normalize
from metric import score 
#from chip_utils.csv_manager import load_gt_chip_csv 

def load_gt_chip_csv(val_chip_csv, scene_id_list):
    ground_truth = pd.read_csv(val_chip_csv, index_col=False)
    ground_truth = ground_truth[ground_truth["scene_id"].isin(scene_id_list)] #.reset_index()
    score_all = False # By default we only score on high and medium confidence labels
    if not score_all:
        ground_truth = ground_truth[ground_truth['confidence'].isin(['HIGH', 'MEDIUM'])].reset_index()
    return ground_truth

SHORELINE_VALIDATION_PATH = '/mnt/omreast_users/phhale/open_ds/xview3/datasets/shoreline/validation'

def submission_score(val_label_file_csv,inference):
    ground_truth = load_gt_chip_csv(val_label_file_csv,inference['scene_id'].unique())
    out = score(inference, ground_truth, shore_root=SHORELINE_VALIDATION_PATH,distance_tolerance=200, shore_tolerance=2)

    print(out)

    """
    aggregate_f(
    loc_fscore (float): F1 score for overall maritime object detection
    length_acc (float): Aggregate percent error for vessel length estimation
    vessel_fscore (float): F1 score for vessel vs. non-vessel task
    fishing_fscore (float): F1 score for fishing vessel vs. non-fishing vessel task
    loc_fscore_shore (float): F1 score for close-to-shore maritime object detection
    """
    return out


if __name__ == '__main__':
    ## load the csv inference file    
    inference_csv = '/mnt/omreast_users/phhale/open_ds/xview3/experiments/yolact/112921_yolact_r50_xView3_npy200_v1_bg/50k_590dd08f71056cacv.csv'
    inference = pd.read_csv(inference_csv)

    val_label_file_csv = '/mnt/omreast_users/phhale/open_ds/xview3/datasets/labels/validation.csv'
    submission_score(val_label_file_csv, inference)

