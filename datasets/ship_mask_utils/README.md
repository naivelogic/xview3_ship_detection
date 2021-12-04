# Quick Start 



## Create Pixel-wise mask segmentation 

- Simple _binary_ morphology operation for image processing. 
  
```
CHIPS_PATH=/mnt/mnt_xview3/datasets/val_chip200_sets/val_590dd08f71056cacv/
CHIP_CSV=/mnt/mnt_xview3/datasets/val_chip200_sets/val_590dd08f71056cacv/val_chip_annotations.csv
COCO_DIR=/mnt/mnt_xview3/datasets/coco_ds/
COCO_JSON_NAME=val_590dd08f71056cacv_coco
python main.py --chips_path ${CHIPS_PATH} --chip_csv ${CHIP_CSV} --ann_out_dir ${COCO_DIR} --ann_file_name ${COCO_JSON_NAME}
```