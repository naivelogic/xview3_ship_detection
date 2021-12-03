import os
from pathlib import Path
import json

class cocoManager:
    def __init__(self):
        #self.bbox_data = []
        #self.mask_data = []
        self.inp_scene_index = None
        self.annotation_id = 1
        self.image_id = 1
        self.is_crowd = 0
        self.images=[]
        self.annotations = []


    def coco_ann_create(self,det,anny): #seg,gbox, gbox_area
        #global image_id, annotation_id, annotations
        ## older no longer needed - used to be with rresult
        #xyxy_bbox = [anny['xyxy_bbox'][1],anny['xyxy_bbox'][0],anny['xyxy_bbox'][3],anny['xyxy_bbox'][2]]
        annotation_info = {
                'det_name': det.name,
                'detect_id': str(det.detect_id),
                'chip_index': det.chip_index,
                'scene_id': det.scene_id,
                'area': anny['area'],   #gbox_area, #int(prop.area),
                'ship_length':anny['ship_length'],
                'iscrowd':0, #null need to update
                'bbox': anny['bbox'],#[rresult[0]], #[s_bbox[0], s_bbox[1], (s_bbox[2]-s_bbox[0]), (s_bbox[3]-s_bbox[1])],
                'segmentation': [anny['segmentation']],#[seg],#[s_seg],
                'image_id': self.image_id,
                'category_id': det.vessel_class,
                'id': self.annotation_id,
                'xyxy_bbox':anny['xyxy_bbox'], #xyxy_bbox, #anny['xyxy_bbox'],
                'xmin':anny['xyxy_bbox'][0], #xyxy_bbox[0], #anny['xyxy_bbox'][1],
                'ymin': anny['xyxy_bbox'][1],#xyxy_bbox[1],#anny['xyxy_bbox'][0],
                'xmax':anny['xyxy_bbox'][2], #xyxy_bbox[2],#anny['xyxy_bbox'][3],
                'ymax':anny['xyxy_bbox'][3], #xyxy_bbox[3],#anny['xyxy_bbox'][2],
        
            }
        self.annotations.append(annotation_info)
        self.annotation_id += 1
            
        #return annotation_id,annotations


    def coco_image_create(self):
        #global output_dir, image_id, chip_npy_path
        #im_pil = Image.fromarray(array *255).convert("L")
        #img_path = Path.joinpath(output_dir,"images/{:06d}.png".format(image_id))
        #im_pil.save(img_path)

        #img_size = fil.size
        #img_size = im_pil.size
        new_img={}
        new_img["license"] = 0
        new_img["file_name"] = self.chip_npy_path 
        #new_img["file_name"] = img_path.name
        new_img["width"] = 200 #img_size[0]
        new_img["height"] = 200 #img_size[1]
        new_img["id"] = self.image_id
        self.images.append(new_img)
        self.image_id += 1

    