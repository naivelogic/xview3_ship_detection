"""
Here just a script check if bbox from the `sub_chip_coco_mask_loop` 
have overlaping boxes shown as %%

https://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html#nms
"""

import numpy as np

def box_area(arr):
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height

def _box_inter_union(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    #union 
    union = area1 + area2 - intersection
    return intersection, union

def box_iou(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    assert(arr1[:, 2:] > arr1[:, :2]).all()
    assert(arr2[:, 2:] > arr2[:, :2]).all()
    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    return iou

def box_iou_checker(box_list):
    for xi, ant in enumerate(box_list):
        if xi == 0:
            # first one just compares against nothing
            continue
        for x in box_list[:xi]:
            is_dup = box_iou(np.array([ant]), np.array([x]))
            if is_dup >= 0.3: #print("!! ERROR Duplicate BBOX (iou > 0.3)")
                return False
    return True