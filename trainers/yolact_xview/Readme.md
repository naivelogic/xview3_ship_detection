## Yolact on xview for ship segmentation

__You Only Look At CoefficienTs__

> Based off of: https://github.com/dbolya/yolact
> 
> You Only Look At CoefficienTs
> ```
>    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
>    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
>     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
>      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
>       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
>       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
>```
>
>A simple, fully convolutional model for real-time instance segmentation. This is the code for our papers:
> - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
> - [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)


## Train Model

Here are the initial procedures for training a yolact model. This initial approach utilizes converted `.png` images and a coco annotated `.json` file. 

```
tmux
cd repos/yolact/
source ~/anaconda3/bin/activate; conda activate yolact-env

mkdir /mnt/omreast_users/phhale/open_ds/xview3/experiments/yolact/tiny/110221_yolact_r50_tiny_200_x0

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1


FOLDER=/mnt/omreast_users/phhale/open_ds/xview3/experiments/yolact/trainv1/110221_yolact_r50_train03_200_x0
python train.py --config=yolact_r50_xview3_x01_config --save_folder ${FOLDER} --log_folder ${FOLDER} \
                --pretrained_models weights/resnet50-19c8e357.pth --batch_size=8
```

Initial training results after 25 epochs on tiny dataset:

```
       |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
   box |  7.76 | 24.76 | 22.16 | 14.70 |  8.10 |  3.60 |  2.35 |  1.51 |  0.36 |  0.01 |  0.00 |
  mask |  8.23 | 21.03 | 17.70 | 15.69 | 13.47 |  7.51 |  5.06 |  1.45 |  0.36 |  0.00 |  0.00 |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
```


## Inferance Image Scene (e.g., image folder)

Steps used for xView Yolact inference pipeline:
- Image Test loop for the DNN predictions
- Each detection outputs boxes/labels/score
- Update the chip col and row for detection if > confidence score
- Calculate the ship length
- Calculate chip offset “coord.json” that maps the pred chip col/row to scene col/row
- Finalize submission csv to challenge requirements

Below are the steps used to perform inference on a scene in the xView dataset (can be any scene including public).

```
source ~/anaconda3/bin/activate tron
TEST_IMGS=/home/redne/xview_dev/datasets/xview_coco/sample/images

FOLDER=/mnt/omreast_users/phhale/open_ds/xview3/experiments/yolact/trainv1/110221_yolact_r50_train03_200_x0
ML=yolact_r50_xview3_x01_8_10000.pth
DS=xview_ds_train03 
CONFIG=yolact_r50_xview3_x01_config
TEST_IMGS=/mnt/omreast_users/phhale/open_ds/xview3/datasets/tiny/coco_chips_200/validation/images/
OUT_DIR=/home/redne/xview_dev/trainers/xview_tron/yolact/10k_sample_imgs
python /home/redne/repos/yolact/eval.py --trained_model=${FOLDER}/${ML} \
               --config=${CONFIG} --score_threshold=0.4 \
               --images=${TEST_IMGS}:${OUT_DIR} --dataset=${DS} --max_images 5
```

Below is an example of xview validation detections from the initial yolact ship detector on a few images:

<img src=media/yolact_initial_val_detections_110321.png height=150>



## Coco evaluation

To run the coco evaluations, below are the procedures. 

```
source ~/anaconda3/bin/activate tron
cd ~/xview_dev/trainers/yolact_xview/coco_evals
python yolact_coco_eval_boxes.py

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.101
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.204
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.069
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.101
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.136
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.161
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.161
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
```


## Latest Experiments week of 11/20 (10 days out from xView3 Competition close)

__Results on `112121_yolact_r50_xView3_npy200_v1` tested only on validation set `590dd08f71056cacv`:__

> TODO: need to kick off leaderboard testing set on public dataset from xView3

| step | loc<br>     fscore | l-fscore<br>     score | vessel<br>     fscore | fishing<br>     fscore | length<br>     acc | aggregate |
|------|--------------------|------------------------|-----------------------|------------------------|--------------------|-----------|
| 10k  |        0.691       |          0.145         |         0.779         |          0.471         |        0.000       |   0.332   |
| 20k  |        0.636       |          0.121         |         0.813         |          0.541         |        0.230       |   0.346   |
| 30k  |        0.682       |          0.088         |         0.825         |          0.537         |        0.241       |   0.367   |
| 40k  |        0.756       |          0.149         |         0.866         |          0.471         |        0.248       |   0.413   |
| 50k  |        0.696       |          0.125         |         0.838         |          0.308         |        0.243       |   0.350   |
| 60k  |        0.723       |          0.126         |         0.860         |          0.414         |        0.244       |   0.382   |

path to model results: `/mnt/omreast_users/phhale/open_ds/xview3/experiments/yolact/112121_yolact_r50_xView3_npy200_v1/60k_590dd08f71056cacv.json`