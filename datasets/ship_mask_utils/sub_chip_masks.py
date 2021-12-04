from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage import measure

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from shapely.geometry import Polygon

from utils.check_bbox_overlap import box_iou_checker
from utils.pixel_masks_utils import im_crop_around, closeContour, binary_mask_filter, ray_tracing_method

def sub_chip_coco_mask_loop(array,detects,SUB_CROP_SIZE = 20):
    bbox_seg_result = dict()
    im2 = Image.fromarray(array * 255).convert("L")
    box_list = list()

    ## Annotation loop (need to add coco annotaitons in this loop)
    for det_ix, det in detects.iterrows():    
        det_col_row = (det.columns, det.rows)
        nbox = im_crop_around(im2, det.columns, det.rows, SUB_CROP_SIZE,SUB_CROP_SIZE)
        cropped = im2.crop(nbox) # left, upper, right, lower
        coco_xywh_bbox, seg,coco_bbox, gbox_area, ship_length = sub_chip_zoom(det_col_row, cropped, nbox)
        if (not coco_xywh_bbox) | (gbox_area < 5):
            return False
        bbox_seg_result[det_ix] = { 'bbox':coco_xywh_bbox,
                                    'segmentation':seg,
                                    'area':gbox_area,
                                    'xyxy_bbox':coco_bbox,
                                    'ship_length':ship_length,}
        
        box_list.append(coco_bbox)
    
    # create some verification before createing coco annotations 
    ## 11/19 done its in the sub_chip_zoom_debug
    ## check for duplicates
    if len(box_list)> 1:
        is_dup_boxes = box_iou_checker(box_list)
        if is_dup_boxes != True:
            return False
    
    return bbox_seg_result
    



def sub_chip_zoom(det_col_row, cropped, nbox):
    override = None
    fil, override = binary_mask_filter(np.array(cropped),override)
    lb = label(np.array(fil), background=255) #label(np.array(fil), background=0)
    bou = mark_boundaries(np.array(cropped), lb)
    props_ = regionprops(lb)
    if len(props_) > 1:
        for a_, prop1 in enumerate(props_):
            ship_coords = np.array([[int(i+nbox[0]), int(j+nbox[1])] for i,j in prop1.coords])
            approx_pgon = measure.approximate_polygon(ship_coords,2)
            s11 = ray_tracing_method(det_col_row[0],det_col_row[1],approx_pgon)
            if s11:
                prop = prop1
                break
            else:
                props_.pop(a_)
                prop = False
    else: 
        prop = True

    # transform sub chip bbox to chip bbox for coco annotaiton
    if (len(props_) == 1) & (prop != False): 
        prop = props_[0]

        ### NEW create mask segmentaiton!!!! 
        ## Get segmentation update with proper coordinates
        bmask = np.pad(fil,pad_width=1, mode='constant', constant_values=255)

        contours = measure.find_contours(bmask,0.5,fully_connected='high')
        contours = np.subtract(contours, 1)
        
        tmpts = contours[0]
        tmpts[:,0] = tmpts[:,0]+ nbox[1]
        tmpts[:,1] = tmpts[:,1]+ nbox[0]
        contour = tmpts
        
        contour = closeContour(contour)
        contour = measure.approximate_polygon(contour, 1)

        if len(contour)<3:
            return False, False, False, False, False 
            
        contour = np.flip(contour,axis =1)
        seg1 = contour.ravel().tolist()
        s_seg = [0 if i < 0 else int(i) for i in seg1] 

        ## Ship / Bbox Area Calc
        plygon = Polygon(np.squeeze(contour))
        area = plygon.area #int(prop.bbox_area) 
        ship_length = plygon.length
        xyxy_bbox = [0 if i < 0 else int(i) for i in list(plygon.bounds)] # minx, miny, maxx, maxy
        coco_xywh_bbox = [xyxy_bbox[0], xyxy_bbox[1], (xyxy_bbox[2]-xyxy_bbox[0]), (xyxy_bbox[3]-xyxy_bbox[1])] # xmin, ymin, width, height
        
    else:
        #print("!!! ERROR MORE THAN 1 CHIP DETECTED !!!")
        return False, False, False, False, False 
    return coco_xywh_bbox, s_seg, xyxy_bbox, area, ship_length
