# xView3 Challenge Installation Notes

## Overall Project Structure

This project utilizes Azure blob containers for cloud storage and mount using `blobfuse`. The folder strcture is described below:

```
├── /mnt/mnt_xview3
│   ├── datasets  
│   │   ├── coco_ds
│   │   ├── labels
│   │   │   ├── train.csv
│   │   │   └── validation.csv
│   │   ├── train
│   │   │   ├── <scene_id> (.e.g, `095f2289c23951e5t`)
│   │   │   │   ├── bathymetry.tif
│   │   │   │   ├── owiMask.tif
│   │   │   │   ├── owiWindDirection.tif
│   │   │   │   ├── owiWindQuality.tif
│   │   │   │   ├── owiWindSpeed.tif
│   │   │   │   ├── __VH_dB.tif__
│   │   │   │   └── __VV_dB.tif__
│   │   │   └── ...
│   │   ├── validation
│   │   ├── public
│   │   ├── shoreline
│   │   │   ├── train
│   │   │   │   ├── <scene_id>_shoreline.npy
│   │   │   │   └── ...
│   │   │   └── validation
│   │   ├── train_chip
│   │   │   ├── <scene_id> (.e.g, `095f2289c23951e5t`)
│   │   │   │   ├── bathymetry
│   │   │   │   │   ├── <chip_id>_bathymetry.npy
│   │   │   │   │   └── ...
│   │   │   │   ├── vh
│   │   │   │   │   ├── <chip_id>_vh.npy
│   │   │   │   │   └── ...
│   │   │   │   ├── vv
│   │   │   │   │   ├── <chip_id>_vv.npy
│   │   │   │   │   └── ...
│   │   │   │   └── coords.json
│   │   │   ├── ...
│   │   │   ├── data_means.npy
│   │   │   ├── data_std.npy
│   │   │   └── train_chip_annotations.csv
│   │   ├── val_chip
│   │   ├── public_chip
│   ├── experiments
│   │   ├── yolact
└───└───└── detectron2 (tbd)
```

## Downloading xView3: Dark Vessels challenge dataset

Download data from the xView3 site: https://iuu.xview.us/download-links, first you'll have to create an account. 

- download and install aria2 to speed up the download: https://aria2.github.io/ by: `sudo apt  install aria2`
- downlod the image.txt file from https://iuu.xview.us/download-links
- download train, validation, public, shoreline (train/validation) and labels (`train.csv`/`validation.csv`)

Below is an sample for downloading the xView3 challenge datasets
```
sudo apt  install aria2
INPUT=<location of the xView3 download text file (e.g., public.txt)>
DIR=/mnt/mnt_xview3/datasets/public
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false

## --dry-run=true #test run

cd /mnt/mnt_xview3/datasets/public
for file in *.tar.gz; do tar xzvf "${file}" && rm "${file}"; done
```


## xView3 Reference Environment Setup

clone: https://github.com/DIUx-xView/xview3-reference

```
conda env create -f environment.yml
conda activate xview3
pip install ipykernel
python -m ipykernel install --user --name xview3


## need to also install the right cuda 11 for pytorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit=11.0
pip install scikit-image 
```
