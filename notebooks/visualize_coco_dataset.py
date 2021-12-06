# %%
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../") # go to parent dir
import numpy as np
import time
from PIL import Image

from datasets.coco_tools.coco_vis_utils import CocoLikeDataset, display_top_masks
## todo run image_from_chip below
#from chip_utils.vh_chip_loader import img_from_chip_v2

# %%
dataset_train = CocoLikeDataset()
MNT_JSON = '/mnt/mnt_xview3/datasets/coco_ds/xview3_ds_mnt/val_set_01_HM_fullnpypath_112021.json'
MNT_IMG = '/mnt/' #'/mnt/mnt_xview3/datasets/'
dataset_train.load_data(MNT_JSON, MNT_IMG)

dataset_train.prepare()
print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# %%
dataset = dataset_train
image_ids = np.random.choice(dataset.image_ids, 2)
for image_id in image_ids:
    #image = dataset.load_image(image_id) # if using .png
    image = img_from_chip(dataset.image_info[image_id]['path']) # .npy version load_npy_xview_image
    mask, class_ids = dataset.load_mask(image_id)
    display_top_masks(image, mask, class_ids, dataset.class_names)

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb #rgb2gray
from PIL import Image

# %%
def img_from_chip(img_path):
    
    data = np.load(img_path)
    data[data < -50] = -50 #process vh and vv channels
    # (default) Puts values b/t 0 and 1, as expected by Faster-RCNN implementation
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    array = np.array(data)
    
    # https://note.nkmk.me/en/python-pillow-putalpha/
    # https://stackoverflow.com/questions/18522295/python-pil-change-greyscale-tif-to-rgb
    im_pil = Image.fromarray(array *255).convert("L")
    rgbimg = Image.new("RGB", im_pil.size)
    rgbimg.paste(im_pil)
    #rgbimg = gray2rgb(array*255)
    
    return np.array(rgbimg)
    

# %%



