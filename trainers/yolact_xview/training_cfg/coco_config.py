##### 11/21/21

xView3_npy200_v1 = dataset_base.copy({
  'name': 'xView3 Challenge DS with HM enhanced masks dataloander .npy chips',
  'train_info': '/mnt/mnt_xview3/datasets/coco_ds/v1_final/trainval_ALL_HM_fullnpypath_112021.json', #All train/val 40383 imgs
  'train_images': '/mnt/',
  'valid_info': "/mnt/mnt_xview3/datasets/coco_ds/v1_final/val_set_3k_HM_fullnpypath_112021.json", #3k images
  'valid_images': '/mnt/',
  'has_gt': True,
  'class_names': ("fishing", "non_fishing", "other"),
  'label_map': { 1:  1, 2: 2, 3: 3}
})

yolact_r50_xview3_sx200_config = yolact_resnet50_config.copy({
    'name': 'yolact_r50_xView3_npy200_v1',
    'dataset': xView3_npy200_v1,
    'num_classes': len(xView3_npy200_v1.class_names) + 1,
    
    # Training params - 112121_yolact_r50_xView3_npy200_v1
    'max_size': 200,
    'max_iter': 200000,
    'lr_steps': (.35 * 200000, .75 * 200000, .88 * 200000, .93 * 200000),
    'discard_mask_area': -1,

})

