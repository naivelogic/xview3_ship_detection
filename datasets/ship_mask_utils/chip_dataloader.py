import numpy as np

## LOAD IMAGE CHANNEL IN DATASET
def get_image_from_chip(chips_path,scene_id,chip_index):
    # loading only the `vh` channel
    data = np.load(f"{chips_path}/{scene_id}/vh/{int(chip_index)}_vh.npy")
    data[data < -50] = -50 #process vh channels
    # (default) Puts values b/t 0 and 1, as expected by Faster-RCNN implementation
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return np.array(data)

def get_next_chip(chip_idx_list, scene_id, df, chips_path):
    # converting list to iterator
    chip_index = next(chip_idx_list)
    
    array = get_image_from_chip(chips_path,scene_id,chip_index)
    detects = df[(df["scene_id"] == scene_id) & (df["chip_index"] == chip_index)]
    
    return array, detects, chip_index
