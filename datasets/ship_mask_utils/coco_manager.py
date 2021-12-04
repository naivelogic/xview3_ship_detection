import os
from pathlib import Path
import json
from utils.xview_coco_defaults import COCO_INFO, COCO_LICENSES, COCO_CATEGORIES

class cocoManager:
    def __init__(self):
        self.inp_scene_index = None
        self.annotation_id = 1
        self.image_id = 1
        self.is_crowd = 0
        self.images=[]
        self.annotations = []


    def coco_ann_create(self,det,anny): 
        annotation_info = {
                'det_name': det.name,
                'detect_id': str(det.detect_id),
                'chip_index': det.chip_index,
                'scene_id': det.scene_id,
                'area': anny['area'],   
                'ship_length':anny['ship_length'],
                'iscrowd':0, #null need to update
                'bbox': anny['bbox'],
                'segmentation': [anny['segmentation']],
                'image_id': self.image_id,
                'category_id': det.vessel_class,
                'id': self.annotation_id,
                'xyxy_bbox':anny['xyxy_bbox'],
                'xmin':anny['xyxy_bbox'][0], 
                'ymin': anny['xyxy_bbox'][1],
                'xmax':anny['xyxy_bbox'][2], 
                'ymax':anny['xyxy_bbox'][3],
        
            }
        self.annotations.append(annotation_info)
        self.annotation_id += 1


    def coco_image_create(self):
        new_img=dict()
        new_img["license"] = 0
        new_img["file_name"] = self.chip_npy_path 
        new_img["width"] = 200 
        new_img["height"] = 200 
        new_img["id"] = self.image_id
        self.images.append(new_img)
        self.image_id += 1

    def save_coco_json(self):
        ### create COCO JSON annotations
        my_dict = dict()
        my_dict["info"]= COCO_INFO
        my_dict["licenses"]= COCO_LICENSES
        my_dict["images"]=self.images
        my_dict["categories"]=COCO_CATEGORIES
        my_dict["annotations"]=self.annotations

        print("saving annotations to coco as json ")
        output_file_path = os.path.join(self.output_dir,f"{self.annotation_file}.json")
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(my_dict))
        print("coco annotations saved: ", output_file_path)

    def save_annotation_csv(self,gt):
        ## CSV
        custom_val = gt[gt['ph_ann'] == True].reset_index(drop=True)
        custom_val.to_csv(os.path.join(self.output_dir,f"{self.annotation_file}.csv"))
