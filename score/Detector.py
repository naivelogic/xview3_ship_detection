#from eval import Detections
import json, os
import numpy as np
import pycocotools

from skimage.measure import label, regionprops
from skimage import measure

class DetectionsManager:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []
        self.inp_scene_index = None
        self.inp_chip_index = None
        self.chip_path = None
        self.results = []

    def get_chip_offsets_json(self):
        # Getting chip transforms
        chip_coord_json_path = os.path.join(self.chip_path,self.inp_scene_index,"coords.json")
        with open(chip_coord_json_path) as fl:
            self.coords = json.load(fl)

    def chip_coord_mapper(self,pred_chip_rows,pred_chip_columns):
        chip_offset_col, chip_offset_row = self.coords["offsets"][self.inp_chip_index]
        # Adjusting chip pixel preds to global scene pixel preds
        chip_pred_row = pred_chip_rows #inf_val["pred_chip_rows"]#[idx]
        chip_pred_column = pred_chip_columns #inf_val["pred_chip_columns"]#[idx]
        scene_pred_column = chip_pred_column + chip_offset_col
        scene_pred_row = chip_pred_row + chip_offset_row
        return scene_pred_row,scene_pred_column

    def add_bbox(self, category_id:int, bbox:list, segm:np.ndarray):

        ## xView3 calculations
        bb = bbox.cpu().unsqueeze(0) #tensor([[ 10, 107,  14, 111]])

        try:
            ## estimating ship length using mask and ellipse major axis    
            lb = measure.label(segm, background=0)
            prop = regionprops(lb)[0]
            # calculating ship length
            
            y0,x0 = prop.centroid
            #orientation = prop.orientation
            x1 = (x0 + np.cos(prop.orientation) * 0.5 * prop.minor_axis_length)
            y1 = (y0 - np.sin(prop.orientation) * 0.5 * prop.minor_axis_length)
            x2 = (x0 - np.sin(prop.orientation) * 0.5* prop.major_axis_length)
            y2 = (y0 - np.cos(prop.orientation) * 0.5* prop.major_axis_length)
            major_length = np.sqrt((x2 - x0)**2 + (y2 - y0)**2)
            #minor_length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            
            lengths = major_length*2 #minor_length #float(minor_length*2)
            #lengths = prop.major_axis_length * 2
            #print("minor: ", float(minor_length*10))
            #print("major: ", float(major_length*10))
            
        except:
            pix_to_m = 2
            for xmin, ymin, xmax, ymax in bb:
                lengths = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
                lengths = float(pix_to_m) * lengths

        # calc 2 update prediction chip row and col
        pred_chip_col = [
            [int(np.mean([box[0], box[2]])) for box in bb] #output["boxes"]]
            if len(bb) > 0
            else []
                        ][0][0]

                    
        pred_chip_rows = [
                [int(np.mean([box[1], box[3]])) for box in bb] #output["boxes"]]
                #if len(output["boxes"]) > 0
                if len(bb) > 0
                else []
                ][0][0]


        ## Step 10: Chip Transformation
        ### Adjusting chip pixel preds to global scene pixel preds
        scene_pred_row,scene_pred_column = self.chip_coord_mapper(pred_chip_rows,pred_chip_col)

        ## Step 11: set is fishing is vessel
        # classes for classification portion of model
        #BACKGROUND = 0, FISHING = 1, NONFISHING = 2, NONVESSEL = 3
        label = int(category_id)
        is_fishing = label == 1 # FISHING = 1
        is_vessel = label in [1,2] # FISHING = 1, NONFISHING = 2

        ## now create boxes
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]
        
        self.results.append({
            "detect_scene_row":int(scene_pred_row),
            "detect_scene_column":int(scene_pred_column),
            "scene_id":str(self.inp_scene_index),
            "is_vessel":is_vessel,
            "is_fishing":is_fishing,
            "vessel_length_m":float(lengths)})
 
    def dump_boxes_json(self, json_path):

        with open(json_path, 'w') as f:
            #json.dump(self.bbox_data, f)
            json.dump(self.results, f)
            
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops