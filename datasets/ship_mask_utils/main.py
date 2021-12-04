import os, sys, json
import warnings
warnings.filterwarnings("ignore")
import argparse
from pathlib import Path
from tqdm import tqdm

from coco_manager import cocoManager
from chip_dataloader import get_next_chip
from sub_chip_masks import sub_chip_coco_mask_loop
from csv_manager import load_chip_csv

def chip_util_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips_path", type=str, default='', required=False)
    parser.add_argument("--chip_csv", type=str, default="" , required=False)
    parser.add_argument("--ann_out_dir", type=float, default=0.5, required=False)
    parser.add_argument("--ann_file_name", type=bool, default=0, required=False,)
    #args = argparse.Namespace(
    #    chips_path= "/mnt/mnt_xview3train/datasets/train_chip200_sets/set_07",
    #    chip_csv = "/mnt/mnt_xview3train/datasets/train_chip200_sets/set_07/train_chip_annotations.csv",
    #    ann_out_dir = '/mnt/mnt_xview3/datasets/coco_ds/xview3train_ds_mnt/',
    #    ann_file_name = 'train_set_07_HM_112021'
    #)
    return parser

def process_scene_annotations(co,scene_ids,gt,chips_path):
    ## 03 initialize the annotaiton loop for each scene in the chip set
    for scene_id in tqdm(scene_ids):
        chip_index_list = gt[gt.scene_id == scene_id].chip_index.unique() #[4966, 5509]
        chip_idx_list = iter(list(chip_index_list))
        

        ## 04 loop through the chip to get the annotaitons
        for x in range(0,len(chip_index_list)):
            # new chip iter to annotate 
            array, detects, chip_index = get_next_chip(chip_idx_list, scene_id, gt, chips_path)
            chip200_set_path  = chips_path.split('/')[4:] #Path(chips_path).parts[4:] # ('val_chip200_sets', 'set_02')
            co.chip_npy_path = os.path.join(chip200_set_path[0],chip200_set_path[1],scene_id,'vh',f'{int(chip_index)}_vh.npy')

            bbox_seg_result = sub_chip_coco_mask_loop(array,detects)
            if bbox_seg_result == False:
                # if we return a false because (a number of things e.g., annotator returned too many ship, couldnt return the ships, ships were too small, too many ships detected) retry for a sub_crop_size of 10
                #print("rerun for crop size = 10")
                bbox_seg_result = sub_chip_coco_mask_loop(array,detects,SUB_CROP_SIZE = 10)
                if bbox_seg_result == False:
                    # if it fails again (annotator returns False) just move on to the next chip
                    # > if it doesnt return a false, save results to coco and DF
                    continue
            ## 05 save annotations for coco_json and GT_DF_csv 
            # if the sub_chipper results == initial detection GT - save to coco annotation
            if (len(bbox_seg_result) == len(detects)): #check to verify the # of GT instances match we are saving  
                for det_ix, det in detects.iterrows():
                    co.coco_ann_create(det,bbox_seg_result[det_ix]) # update the coco_annotations for each instance
                co.coco_image_create()
            else: 
                print("results not saved")
                continue

    return gt


if __name__ == '__main__':
    #args = chip_util_args().parse_args()
    #val_590dd08f71056cacv
    args = argparse.Namespace(
        chips_path= "/mnt/mnt_xview3/datasets/val_chip200_sets/set_01/",
        chip_csv = "/mnt/mnt_xview3/datasets/val_chip200_sets/set_01/val_chip_annotations.csv",
        ann_out_dir = '/mnt/mnt_xview3/datasets/coco_ds/',
        ann_file_name = 'val_590dd08f71056cacv_coco'
    )
    print("Command Line Args:", args)

    ## 01 initalize the coco dataset for storing coco_annotations
    co = cocoManager()
    co.output_dir = Path(args.ann_out_dir) 
    co.output_dir.mkdir(exist_ok=True)
    co.annotation_file = args.ann_file_name


    ## 02 Load the GT of chip set
    gt = load_chip_csv(args.chip_csv)  # only including high and medium confidence labels
    gt_chip_index_list = gt.chip_index.unique()
    print("# of image chips: ", len(gt_chip_index_list))

    ###  getting the chip scene_ids
    scene_ids = os.listdir(args.chips_path)
    scene_ids.remove(Path(args.chip_csv).name)
    print("Lenth of dataset IDs: ", len(scene_ids))

    gt = process_scene_annotations(co,scene_ids,gt, args.chips_path)

    co.save_coco_json()
    #co.save_annotation_csv(gt)