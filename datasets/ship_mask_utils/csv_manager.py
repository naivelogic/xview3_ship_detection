import os, sys
import pandas as pd
import numpy as np

def load_chip_csv(chip_csv):
    # usage: gt = load_chip_csv(val_chip_csv)
    gt = pd.read_csv(chip_csv,index_col=0)

    ## FILTERS remove any CHIPS where chips that are on the border (overlaps)
    exclude_chips_indx_on_border = np.unique(gt[(gt['rows']<=3) | (gt['columns']<=3) |(gt['rows']>=197) | (gt['columns']>=197)].chip_index)
    gt = gt[~gt["chip_index"].isin(exclude_chips_indx_on_border)] #.reset_index()

    #score_all = True # TRAIN!! need to incorperate low as they are all SHIPS for TRAIN CHIPS DS
    score_all = False # VAL only  and TRAIN for genral DS
    # By default we only score on high and medium confidence labels
    if not score_all:
        print("using H and Med only")
        gt = gt[gt['confidence'].isin(['HIGH', 'MEDIUM'])].reset_index()

    #gt_chip_index_list = gt.chip_index.unique()
    #print("# of image chips: ", len(gt_chip_index_list))


    ## set items for result csv
    gt["xmin"] = 0
    gt["xmax"] = 0
    gt["ymin"] = 0
    gt["ymax"] = 0
    gt['area'] = 0
    gt['anny_id'] = 0
    gt["ph_ann"] = False
    return gt