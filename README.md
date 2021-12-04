# xView3 Challenge


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



## Getting Started

* [Installation_doc](docs/01_Installation.md) instructions
* [Dataset](docs/02_Dataset_Details.md) detail procedures for downloading and preparing the dataset
* To start training [Quickstart.md](docs/03_Quickstart.md)
* [Acknowledgements.md](docs/Acknowledgements.md)
