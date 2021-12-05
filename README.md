# xView3 Challenge

The goal of this project is to detect, classify and estimate the length of dark vessels (e.g., ships) using pixel-wise masks segmentation as a partiipant in the [xView3: Dark Vessels Challenge](https://iuu.xview.us/challenge) the fall of 2021.

Below is an example of xview validation detections from the initial yolact ship detector on a few images:

<img src=trainers/yolact_xview/media/yolact_initial_val_detections_110321.png height=150>


## Solution description

> __Abstract__—This report describes the techniques and experiments for improving automatic ship detection from synthetic aperture radar (SAR) satellite imagery as a participant in the xView3 Dark Vessels Challenge 2021. The xView3 Challenge provides a large multi-dimensional dataset of SAR satellite views to benchmark new approaches to automatically detect illegal fishing activities at a global scale. Computer vision methods and Azure Machine Learning services are utilized in this challenge aiming to advance research contributions in extracting accurate ships mask and dimensions to enable performance improvements in ship detection, ship classification and estimating the length of detected ships. The initial technique has been tested and evaluated on the xView3 Challenge public dataset for benchmarking the performance of the trained models were the proposed method ranked 45 on the leaderboard (before unverified sumbissions were removed). While areas of improvement are reflected, the detector and classifier do not outperform the xView3 reference model for this challenge however the proposed method for estimating the length of ships provided positive results. Visual comparisons of the proposed method for delineating the vessel outline in SAR images using mask segmentation indicated more better ground truths from those provided by experts using manual analysis, however manual expert review still may be needed for verifying the classification of ships. 

updated: at the end of the xView3 challenge our submission was ranked in the top 50, rank 45. This was before xView3 removed the unverified submissions.

* paper draft: `paper/Improve Illegal Ship Detection Using Pixel-Wise Mask.pdf`
* presentation: `paper/xView3 Challenge Paper Summary Presentation.pdf`

--------

## Challenge Description

The [xView3: Dark Vessels Challenge](https://iuu.xview.us/challenge) leaderboard performance is tested on the `public` dataset that does not contain any labels, just scenes containing a set of co-registered SAR images indexed by a unique xView3 scene ID.  

__Submission task:__ For each scene in the `public` xView3 challenge dataset, the trained model is to:
1. identify the maritime objects
2. estimate the length of the object
3. classify it as a `vessel` or `non-vessel`
4. for each `vessel` classify each as `fishing` or `non-fishing`. _(`non-vessel` are assumed to be `non-fishing`)_

__Submission format:__ the xView3 challenge submission format required prediction results to be provided as a `.csv` file with the following headings:
* `scene_id:` (str) the unique ID for the xView3 scene
* `detect_scene_row`: (int) pixel coordinate in the vertical (y) axis
* `detect_scene_column`: (int) pixel coordinate in the horizontal (x) axis
* `is_vessel`: (bool), True if the object is a `vessel`; False otherwise
* `is_fishing`: (bool), True if the object is a `fishing-vessel` and false otherwise
* `vessel_length_m`: (float), estimated length of the vessel, in meters

> source: https://iuu.xview.us/challenge


-----------
## Getting Started

* [Installation_doc](docs/01_Installation.md) instructions
* [Dataset](docs/02_Dataset_Details.md) detail procedures for downloading and preparing the dataset
* To start training [Quickstart.md](docs/03_Quickstart.md)
* [Acknowledgements.md](docs/Acknowledgements.md)

## Data Preparations

Download and extract dataset from the xView3 site: https://iuu.xview.us/download-links, first you'll have to create an account. 

Preprocessing is done by runing a couple of scripts: 

__1. Split raw SAR Images into image chips__

Run `python datasets/split_images.py`. This script will split the images into smaller sizes that can be used for DNN trainings. _update script params before running_

__2. Generate Masks__

Run `python datasets/ship_mask_utils/main.py`. This script will generate pixel masks for each of the marinetime objects in chip annotation .csv. The output of this script is a coco_annotation.json file with segmentations. 

## Training

Training yolact instance segmentation is implemented for this challenge solutions. Install yolact as defined in the submodule: `repos/yolact`. The scripts and model configuration is defined in `trainers/yolact_xview`.

During training checkpoints are saved every 10 epochs. 

### AzureML and Training Dockerfile

> TODO: need to add to repo

The training environment for running yolact/detectron2 models are built as docker images. Refer to the submodule: `repos/azureml_cv` for details regarding the yolact/detectron2 Dockerfiles and samples used for kicking of training experiment in AzureML.


## Hardware Requirements

Most training and edevelopment was on an Azure VM configured with 1-2 Nvidia Tesla K80 GPUs. Overall training requires at least 2 GPUs with 12GB memory. Batch size should be adjusted according to the number of GPUs. 

> TODO: experiment with V100 gpus to evaludate if performaance increases if we utilized a larger batch size.

## Inference

To test the model on a new scene or folder of scenes run `test_model_on_folder.py`

## Submission

__Create Submission__: TODO


__view submission__: 




