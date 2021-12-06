#source ~/anaconda3/bin/activate  tron
import sys, os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from score.Detector import DetectionsManager
from tqdm import tqdm
from chip_utils.csv_manager import load_gt_chip_csv
from chip_utils.dev.image_from_chip import img_from_chip_v2 as img_from_chip

#sys.path.append("/home/redne/repos/yolact/") # go to parent dir
sys.path.append("../repos/yolact/") # go to parent dir
from yolact import Yolact
from data import set_cfg, set_dataset
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess

##### Setup ##### 
torch.backends.cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def yolact_predict(args):
    ### Load Yolact Mdoel
    from data import cfg, set_cfg, set_dataset

    set_cfg(args.TRAIN_CFG)
    set_dataset(args.DS_NAME)
    net = Yolact().cuda()
    net.load_weights(args.ML_FULL_PATH)
    net.eval()
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    ## initialize detection manager
    detections = DetectionsManager()
    annotation_id = 1
    detections.chip_path = args.chip_path

    for scene_id in args.scene_ids:
        print("processing scene: ", scene_id)
        detections.inp_scene_index = scene_id   # set scene_id to get chip/img level ids
        detections.get_chip_offsets_json()      # also load the chip_coord_json gettign the coods for the scene. 

        # used only for Public test (when chip IDs are not known)
        #for chip_id in tqdm(range(0,len(detections.coords['offsets']))):
        for chip_id in tqdm(args.gt_chip_ids):
            detections.inp_chip_index = chip_id                                     #chip_id  to be processes
            in_path = os.path.join(detections.chip_path,scene_id,'vh', str(chip_id)+'_vh.npy') #npy image path

            array = img_from_chip(in_path) # grayscale image array 
            frame = torch.from_numpy(np.array(array)).cuda().float()
            h, w, _ = frame.shape

            ### Predict Image
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = net(batch)
    
            ## Processing Predicted Detections #yolact_postprocess()
            classes, scores, boxes, masks = postprocess(preds, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
            with torch.no_grad():
                if classes.size(0) == 0:
                    continue
                classes = list(classes.cpu().numpy().astype(int))
                boxes = boxes.cuda() # x,y,w_,h_
                masks = masks.view(-1, h, w).cpu().numpy()

                for i in range(masks.shape[0]):
                    # Make sure that the bounding box actually makes sense and a mask was produced
                    # xmin, xmax, ymin, ymax
                    if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                        detections.add_bbox(classes[i]+1, boxes[i,:],segm=masks[i,:,:].astype(np.uint8))
                    annotation_id +=1

    # Save submission results as csv
    from pandas.io.json import json_normalize
    inference = json_normalize(detections.results)
    inference.to_csv(args.SUBMISSION_CSV_PATH)
    print("Submission csv: ", args.SUBMISSION_CSV_PATH)

    ## Validate score predictions
    if args.VALIDATE_SCORE_SUBMISSIONS:
        print("... Validate score predictions ")
        from score.score_pred import submission_score
        val_label_file_csv = '/mnt/omreast_users/phhale/open_ds/xview3/datasets/labels/validation.csv'
        out = submission_score(val_label_file_csv, inference)
    #print(out)

if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(
        fast_nms = True,
        cross_class_nms= False,
        mask_proto_debug=False,
        display_lincomb = False,
        crop = True,
        score_threshold = 0.2,
        top_k = 5,
        web_det_path = None
    )

    MNT_ROOT = '/mnt/omreast_users/phhale/open_ds/xview3/experiments/yolact/'
    ML_DIR = '112921_yolact_r50_xView3_npy200_v1_bg' #'112121_yolact_r50_xView3_npy200_v1' #'112121_yolactpp_r50_xView3_npy200_v1' 'yolactpp_r50_xview3_sx200_112121_7_20000.pth'
    PTH = 'yolact_r50_xView3_npy200_v1bg_19_50000.pth' #11_30000. #'yolact_r50_xView3_npy200_v1_15_40000.pth' #19_50000 15_40000 11_30000 7_20000 3_10000 23_60000 31_80000 27_70000 35_90000 39_100000
    CKPT='50k'
    ML_FULL_PATH = os.path.join(MNT_ROOT,ML_DIR,PTH)
    #args.score_threshold = 0.4 #0.2 or 0.4
    TRAIN_CFG = "yolact_r50_xview3_sx200_config" #'yolactpp_r50_xview3_sx200_config' #
    DS_NAME = "xView3_npy200_v1"

    args.SUBMISSION_CSV_PATH=os.path.join(MNT_ROOT,ML_DIR,f'{CKPT}_val_set02.csv')
    args.ML_FULL_PATH = ML_FULL_PATH
    args.TRAIN_CFG = TRAIN_CFG
    args.DS_NAME = DS_NAME
    args.VALIDATE_SCORE_SUBMISSIONS = True


    args.chip_path = "/mnt/mnt_xview3/datasets/val_chip200_sets/set_02/"
    args.scene_ids = os.listdir(args.chip_path)
    #args.scene_ids = ['590dd08f71056cacv'] #['3ceef682fbe4930av','4da9db72dea50504v','590dd08f71056cacv'] # DEBUG

    ## get Validation chip IDs instead of all of the chips
    val_label_file_csv = '/mnt/mnt_xview3/datasets/val_chip200_sets/set_02/val_chip_annotations.csv'
    gt = load_gt_chip_csv(val_label_file_csv,args.scene_ids)
    args.gt_chip_ids = list(gt.chip_index) 
    print("# of GT chips: ", len(args.gt_chip_ids))

    yolact_predict(args)
