#https://github.com/wimlds-trojmiasto/detect-waste/blob/main/utils/dataset_converter.py
import json

def concatenate_datasets(list_of_datasets, dest=None):
    # concatenate list of datasets into one single file
    # the first dataset in the list will be used as a base
    # and the rest of datasets will be appended
    last_ann_id = 0
    last_im_id = 0
    concat_dataset = {}
    concat_dataset['images'] = []
    concat_dataset['annotations'] = []

    for i, annot in enumerate(list_of_datasets):
        with open(annot, 'r') as f:
            dataset = json.loads(f.read())

        anns = dataset['annotations'].copy()
        images = dataset['images'].copy()

        img_dict = {}
        for im in images:
            img_dict[im['id']] = last_im_id
            im['id'] = last_im_id
            last_im_id += 1

        for ann in dataset['annotations']:
            ann['image_id'] = img_dict[ann['image_id']]
            ann['id'] = last_ann_id
            last_ann_id += 1

        concat_dataset['images'] += images
        concat_dataset['annotations'] += anns

    concat_dataset['info'] = dataset['info']
    concat_dataset['licenses'] = dataset['licenses']
    concat_dataset['categories'] = dataset['categories']


    if dest is None:
        return concat_dataset
    else:
        with open(dest, 'w') as f:
            json.dump(concat_dataset, f)
        print('Saved results to', dest)
        print("last_ann_id: ",last_ann_id, " last_im_id: ", last_im_id)

## v1_final DS 11/21/21
#d = '/mnt/mnt_xview3/datasets/coco_ds/xview3train_ds_mnt/train_sets_06_11_HM_fullnpypath_112021.json' #last_ann_id:  25730  last_im_id:  23054
#d = '/mnt/mnt_xview3/datasets/coco_ds/xview3_ds_mnt/train_sets_01_03_HM_fullnpypath_112021.json' # last_ann_id:  11408  last_im_id:  10094
#d = "/mnt/mnt_xview3/datasets/coco_ds/xview3_ds_mnt/val_set_ALL_HM_fullnpypath_112021.json" # last_ann_id:  9678  last_im_id:  7235
#d = '/mnt/mnt_xview3/datasets/coco_ds/v1_final/trainval_ALL_HM_fullnpypath_112021.json' #last_ann_id:  46816  last_im_id:  40383
#d = '/mnt/mnt_xview3/datasets/coco_ds/v1_final/trainval_ALL_HM_fullnpypath_112021.json' #last_ann_id:  46816  last_im_id:  40383
#d = '/mnt/mnt_xview3/datasets/coco_ds/v1_final/bg_val_3k_112921.json' #last_ann_id:  0  last_im_id:  4000 bg_val_set01_112921 - set3

ds_list = [
    '/mnt/mnt_xview3/datasets/coco_ds/v1_final/trainval_ALL_HM_fullnpypath_112021.json',
    '/mnt/mnt_xview3/datasets/coco_ds/v1_final/bg_val_3k_112921.json'
]

d = '/mnt/mnt_xview3/datasets/coco_ds/v1_final/trainval_BGval3k_ALL_HM_fullnpypath_112921.json' #last_ann_id:  46816  last_im_id:  44383


c = concatenate_datasets(ds_list, dest=d)
