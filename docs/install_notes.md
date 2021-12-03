### Downloading xView3: Dark Vessels challenge dataset

https://iuu.xview.us/download-links

- download and install aria2 to speed up the download: https://aria2.github.io/ by: `sudo apt  install aria2`
- downlod the image.txt file from https://iuu.xview.us/download-links

```
sudo apt  install aria2
INPUT=tiny.txt
DIR=/mnt/omreast_users/phhale/open_ds/xview3/datasets/tiny
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false

INPUT=shoreline.txt
DIR=/mnt/omreast_users/phhale/open_ds/xview3/datasets/shoreline
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false


INPUT=/home/redne/xview_dev/datasets/one_sample/val_1.txt
DIR=/mnt/omreast_users/phhale/open_ds/xview3/datasets/dev/one_sample/validation
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false

--dry-run=true #test run

for file in *.tar.gz; do tar xzvf "${file}" && rm "${file}"; done
```

### Quick Start with xview3

https://github.com/DIUx-xView/xview3-reference

```
conda env create -f environment.yml
conda activate xview3
pip install ipykernel
python -m ipykernel install --user --name xview3


## need to also install the right cuda 11 for pytorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit=11.0


## 10/11 - for processing SAR shipts for demo https://github.com/simon-donike/morphology-ship_detection
pip install scikit-image pyAPRiL

```

### downloading dataset 10/23/21

```
#DIR=/mnt/omreast_users/phhale/open_ds/xview3/datasets/shoreline

cd /home/redne/xview_dev/
INPUT=./datasets/downloads/train.txt
DIR=/mnt/mnt_xview3/datasets/train
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false

cd /mnt/mnt_xview3/datasets/train
for file in *.tar.gz; do tar xzvf "${file}" && rm "${file}"; done

INPUT=/home/redne/xview_dev/datasets/downloads/train.txt
DIR=/mnt/mnt_xview3/datasets/train
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false



INPUT=./datasets/downloads/val.txt
DIR=/mnt/mnt_xview3/datasets/validation
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false

/mnt/mnt_xview3/datasets/validation
for file in *.tar.gz; do tar xzvf "${file}" && rm "${file}"; done


INPUT=./datasets/downloads/public.txt
DIR=/mnt/mnt_xview3/datasets/public
aria2c --input-file=$INPUT --auto-file-renaming=false --continue=true --dir=$DIR --dry-run=false

cd /mnt/mnt_xview3/datasets/public
for file in *.tar.gz; do tar xzvf "${file}" && rm "${file}"; done


```


### annotate new dataset x200

```
source ~/anaconda3/bin/activate xview3
cd /home/redne/xview_dev/notebooks/dataloader/

python 01_create_chips.py

```