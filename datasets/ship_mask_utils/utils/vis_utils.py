import matplotlib.pyplot as plt
import matplotlib.patches as patches

def vis_sub_chip_noGT(cocoJSON, array):
    plt.figure(figsize=(5,5))
    fig, ax = plt.subplots(figsize=(5,5))

    for tj in cocoJSON:
        bb= cocoJSON[tj]['xyxy_bbox'] 
        rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect) # Add the patch to the Axes
    ax.imshow(array)
    ax.set_title("Array Image")


def vis_val_GT(det_df, chip_array):
    tmp_bbox_list = []
    for o, d in det_df.iterrows():
        tmp_bbox_list.append([int(d.top % 200), int(d.left% 200), int(d.bottom % 200), int(d.right % 200)])

    plt.figure(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(10,10))

    for tbox in tmp_bbox_list:
        #print('Found bbox', tbox)    
        rect = patches.Rectangle((tbox[1], tbox[0]), tbox[3]-tbox[1],tbox[2]-tbox[0],  linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect) # Add the patch to the Axes
    ax.imshow(chip_array)
    ax.set_title("Array Image")

def vis_sub_chip_with_valGT(rresult,array,detects):
    
    ## Visualize Results
    fig, ax = plt.subplots(ncols=2, figsize=(10,10))

    ## Plot for submask chip detections
    for tj in rresult:
        
        ## bbox_seg_result
        rect = patches.Rectangle((rresult[tj]['xyxy_bbox'][0], rresult[tj]['xyxy_bbox'][1]), 
                                 rresult[tj]['xyxy_bbox'][2]-rresult[tj]['xyxy_bbox'][0],
                                 rresult[tj]['xyxy_bbox'][3]-rresult[tj]['xyxy_bbox'][1], 
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax[0].add_patch(rect)

    ax[0].imshow(array, cmap='gray')
    ax[0].set_title("New GT Sub Image Chip")
    ax[0].axis('off')

    ## Plot Val GT
    tmp_bbox_list = []
    for o, d in detects.iterrows():
        tmp_bbox_list.append([int(d.top % 200), int(d.left% 200), int(d.bottom % 200), int(d.right % 200)])

    for tbox in tmp_bbox_list:
        #print('Found bbox', tbox)    
        rect = patches.Rectangle((tbox[1], tbox[0]), tbox[3]-tbox[1],tbox[2]-tbox[0],  linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect) # Add the patch to the Axes
    ax[1].imshow(array, cmap='gray')
    ax[1].set_title("Original GT Val Image")
    ax[1].axis('off')
    