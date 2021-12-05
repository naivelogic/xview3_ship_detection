# :anchor: Satellite imagery datasets containing ships.<a name="TOP"></a> 
A list of optical and radar satellite datasets for ship detection, classification, semantic segmentation and instance segmentation tasks.

>source: https://github.com/JasonManesis/Satellite-Imagery-Datasets-Containing-Ships


## :satellite: Radar Satellite Datasets : 
* **SSDD (SAR Ship Detection Dataset) - 2017, Li et al.**  ↦ Detection + Semantic Segmentation 
* **OpenSARship-1.0, 2.0- 2017, Huang et al.** ↦ Detection
* **SAR-Ship-Dataset - 2019, Wang et al.** ↦ Detection  *                              
* **AIR-SARShip -1.0, 2.0 - 2019, Sun et al.** ↦ Detection  *                                        
* **HRSID (High-Resolution SAR Images Dataset) - 2020, Wei et al.** ↦ Detection + Instance Segmentation
* **LS-SSDD-v1.0 (Large-Scale SAR Ship Detection Dataset) - 2020, Zhang et al.** ↦ Detection
* **FUSAR-Ship1.0 - 2020, Hou et al.** ↦ Classification 
* **SSDD (SAR Ship Detection Dataset) - 2021, Zhang et al.** ↦ Detection + Instance Segmentation 

## :eyes: Optical Satellite Datasets : 
* **HRSC2016 (High Resolution Ship Collection 2016) - 2016, Liu et al.** ↦ Detection + Instance Segmentation 
* **Ships in Satellite Imagery Dataset - 2017, Kaggle** ↦ Detection as Classification
* **Airbus Ship Detection Challenge Dataset - 2018, Kaggle** ↦ Detection  
* **xView Dataset  - 2018, Lam et al.** ↦ Detection  
* **DOTA (Dataset for Object deTection in Aerial images) - 2018, Xia et al.** ↦ Detection   
* **TGRS-HRRSD (High-Resolution Remote Sensing object Detection) - 2018, Kaggle** ↦ Detection
* **MASATI-v2 (MAritime SATellite Imagery dataset) - 2018,  Gallego et al.** ↦ Detection
* **DIOR(object Detection In Optical Remote sensing images) - 2019, Li et al.** ↦ Detection  *   
* **FGSD (Fine-Grained Ship Detection) - 2020, Chen et al.** ↦ Detection  *   
* **PSDS (Peruvian Ship Data Set) + MSDS (Mini Ship Data Set) - 2020, Cordova et al.** ↦ Detection  *                                 

*The specific datasets could not be accessed.    

- - - -
<p align="center">
    <b>Radar Satellite Datasets</b>
</p>

- - - -

### SSDD (SAR Ship Detection Dataset)
* It consists of 1160 SAR images with average dimensions of 500×500 pixels.
* This specific dataset includes 2358 ship instances.
* The spatial resolutions of SAR images are from 1 to 15 meters per pixel.
* These 1160 images were obtained from RadarSat-2, TerraSAR-X and Sentinel-1 satellites.
* The above 1160 images is are in .jpeg format with 24 bit color depth (8 bits per channel).
* Dataset images have mixed HH, HV, VV, and VH polarizations.
* Every image corresponds to an .json file which contains each ship’s pixel-based segmentation.
* Paper Link: https://ieeexplore.ieee.org/document/8124934 
* Dataset Link: https://drive.google.com/file/d/1grDw3zbGjQKYPjOxv9-h4WSUctoUvu1O/view

### OpenSARship-1.0
* OpenSARShip 1.0 is a medium-resolution ship dataset consisted of 11346 image chips cropped from a total of 41 Sentinel-1 images.
* It covers mainly 5 ports in Asia and it  has 17 types (AIS types) of ships in total.
* The spatial resolutions of the images are 2.7 × 22 to 3.6 × 22 and 20 × 22 meters.
* Dataset images have mixed  VV and VH polarizations.
* Each ship image corresponds to  an automatic identification system (AIS) message.
* The detailed information of each ship chip, containing the AIS messages, the SAR ship signatures, and the messages provided by the MarineTraffic Website, is listed in an XML file named Ship.xml.
* The image sizes range from 30 × 30 to 120 × 120 pixels.
* For every Sentinel-1 SAR image, four subfolders provide four formats of ship chips: original data, visualized data in greyscale, visualized data in pseudo-color, and calibrated data.
* Original and calibrated data are in .tiff format and have 128 bit color depth (32 bits per channel).
* Visualized data in pseudo-color have 24 bit color depth (8 bits per channel) and are in .png format.
* Visualized data in greyscale have 8 bit color depth and are in .tiff format.
* Paper Link: https://ieeexplore.ieee.org/document/8067489 
* Dataset Link: http://opensar.sjtu.edu.cn/Data/Search 

### OpenSARship-2.0
* OpenSARShip 2.0 consists of 34528 image chips cropped from a total of 87 Sentinel-1 images.
* The whole products are in the interferometric wide swath (IW) mode.
* The OpenSARShip contains two available products of the IW mode: the single look complex (SLC) and the ground range detected (GRD) products.
* These 87 Sentinel-1 images, 52 from GRD and 35 from SLC imageries, are selected from 10 typical intense marine traffic scenes globally in latest years.
* For Sentinel-1 the spatial resolutions of the images 2.7 × 22 to 3.6 × 22 and 20 × 22 meters.
* Dataset images have mixed  VV and VH polarizations.
* Each ship image corresponds to an automatic identification system (AIS) message.
* The detailed information of each ship chip, containing the AIS messages, the SAR ship signatures, and the messages provided by the MarineTraffic Website, is listed in an XML file named Ship.xml
* The image sizes range from 30 × 30 to 120 × 120 pixels.
* For every Sentinel-1 SAR image, four subfolders provide four formats of ship chips: original data, visualized data in greyscale, visualized data in pseudo-color, and calibrated data.
* Paper Link: https://ieeexplore.ieee.org/document/8124929 
* Dataset Link: http://opensar.sjtu.edu.cn/Data/Search 

### SAR-Ship-Dataset 
* It consists of 43819 ship chips of 256 × 256 pixels.
* The specific dataset includes 59535 ship instances.
* This dataset was created using 102 Chinese Gaofen-3 images and 108 Sentinel-1 images.
* For Gaofen-3, the images have spatial resolutions of 3 m, 5 m, 8 m, 10m, and 25 m per pixel and for Sentinel-1 the spatial resolutions of the images are 1.7 × 4.3 to 3.6 × 4.9 and 20 × 22 meters.
* For Gaofen-3 the imaging modes are Ultrafine Strip-Map (UFS), Fine Strip-Map 1 (FSI), Full Polarization 1 (QPSI), Full Polarization 2 (QPSII), and Fine Strip-Map 2 (FSII).
* For Sentinel-1, the imaging modes are S3 Strip-Map (SM), S6 SM, and IW-mode.
* Each ship chip corresponds to an Extensible Markup Language (XML) file, indicating the ship location, the ship chip name, and the image shape.
* Paper Link: https://www.mdpi.com/2072-4292/11/7/765 
* Dataset Link: https://github.com/CAESAR-Radi/SAR-Ship-Dataset 

### AIR-SARShip-1.0 
* The AIR-SARShip-1.0 dataset is collected from the Gaofen-3 satellite, and contains 31 images of large scenes.
* The imaging mode has both spotlight and strip modes. All images are in single polarization mode with a size of about 3000 × 3000 pixels.
* The raw SAR images (3000 × 3000 pixels) cropped into 500 × 500 pixel sub-images.
* The above 6×6×31 = 1116 sub-images are in .tiff format.
* Every image corresponds to a XML label file which includes image file name, pixel size, number of channels, resolution, category, and position of each target box.
* The spatial resolutions of SAR images are 1 and 3 meters per pixel.
* Paper Link: http://radars.ie.ac.cn/en/article/doi/10.12000/JR19097 
* Dataset Link: http://radars.ie.ac.cn/web/data/getData?dataType=SARDataset_en&pageType=en 

### AIR-SARShip-2.0 
* Dataset is collected from the Gaofen-3 satellite, and contains 300 images of large scenes.
* The imaging mode has both spotlight and strip modes.
* All images are in single polarization mode. 
* In dataset, each image size is about 1000×1000 pixels, with .tiff format, single channel, 16-bit image depth.
* The spatial resolutions of SAR images are 1 and 3 meters per pixel.
* Each image corresponds to an .xml file, provides the detail information, including the length and width dimensions, the category and the position of the target box.
* Paper Link: http://radars.ie.ac.cn/en/article/doi/10.12000/JR19097 
* Dataset Link: http://radars.ie.ac.cn/web/data/getData?dataType=SARDataset_en&pageType=en 

### HRSID (High-Resolution SAR Images Dataset)
* The specific dataset contains 116 co-polarized and 20 cross-polarized SAR imageries.
* The original imageries for constructing HRSID are 99 Sentinel-1B imageries, 36 TerraSAR-X and 1 TanDEM-X imageries.
* The above 136 panoramic SAR imageries cropped to 5604 high-resolution SAR images.
* These 5604 images have dimensions of 800 × 800 pixels, resolution of 96 dpi, and there are in .jpeg format.
* The colour depth of the images is 8 bits (one channel). 
* The extracted 5604 high-resolution SAR images contain 16951 ship instances.
* The spatial resolutions of SAR images are 0.5, 1 and 3 meters per pixel.
* The annotations of each instance are the corresponding bounding box and the ship’s outline. 
* The annotations of each SAR image constitute a .json file in MS COCO dataset format.
* Paper Link: https://ieeexplore.ieee.org/abstract/document/9127939
* Dataset Link: https://github.com/chaozhong2010/HRSID 

### LS-SSDD-v1.0 (Large-Scale SAR Ship Detection Dataset)
* It consists of 15 large-scale SAR images, obtained from Sentinel-1 satellite, with size of 24000 × 16000 pixels.
* These 15 large-scale SAR images were cut into 9000 sub-images with dimensions of 800 × 800 pixels.
* The above 9000 images are in .jpeg format.
* Spatial resolution: No information available.
* Every image has a corresponding .xml file which contains one bounding box for each one of image’s instances.
* Dataset images have VV and VH polarizations.
* Paper Link: https://www.mdpi.com/2072-4292/12/18/2997/html 
* Dataset Link: https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN 

### FUSAR-Ship Dataset v1.0 
* FUSAR-Ship dataset has a total of 15 ship categories, 98 ship subcategories, consisted of 126 GF-3 scenes covering various scenarios.The imaging mode of those 126 images is ultrafine strip-map (UFS) mode.
* It includes over 5000 ship chips with AIS messages  and some other types of marine targets and background clutters.
* The above single-band ship images have dimensions of 512 × 512 pixels.
* These images are in .tiff format and their color depth is 8 bits.
* These image chips are stored under subfolder named after its 'category/subcategory'. The filename of each sample follows the convention: Ship_CxxSyyNzzzz.tiff, where xx is the index of category, yy is the index of subcategory and zzzz is the index of this particular sample.
* Dataset images have VV and HH polarizations.
* The matchup meta data is compiled in the file 'meta.csv' or 'meta.xls', which follow the format of: id mmsi length width polarMode centerLookAngle heightspace widthspace path.
* Paper Link: https://link.springer.com/article/10.1007/s11432-019-2772-5 
* Dataset Link: http://www.emwlab.fudan.edu.cn/resources/main.psp 

### SSDD (SAR Ship Detection Dataset) 2021
* It consists of 1160 SAR images with dimensions of 500×350 pixels.
* This specific dataset includes 2358 ship instances.
* The spatial resolutions of SAR images are from 1 to 15 meters per pixel.
* These 1160 images were obtained from RadarSat-2, TerraSAR-X and Sentinel-1 satellites.
* The above 1160 images is are in .jpeg format with 24 bit color depth.
* Dataset images have mixed HH, HV, VV, and VH polarizations.
* In the new version of SSDD, three kinds of annotational information are provided: 
  *  Horizontal bounding box.
  *  Rotated bounding box.
  *  Pixel-based segmentation.
* The above annotations are in Pascal VOC and MS COCO dataset formats, exept of rotated bounding boxes, which are only available on Pascal VOC format.
* Paper Link: https://www.mdpi.com/2072-4292/13/18/3690 
* Dataset Link: https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view?usp=sharing 

- - - -
<p align="center">
    <b>Optical Satellite Datasets</b>
</p>

- - - -

### HRSC2016 (High Resolution Ship Collection 2016)
* Dataset contains 1061 images which derived from Google Earth. 
* The above 1061 images including 70 sea images with 90 samples and 991 sea-land images with 2886 samples.
* The image spatial resolutions are between 0.4 m and 2 m. 
* The image sizes range from 300 × 300 to 1500 × 900 and most are larger than 1000 × 600 and are in .bmp format.
* The colour depth of the above 1061 images is 24 bit (8 bits per channel).
* In HRSC2016 dataset, three kinds of bounding information are provided: 
  *  Horizontal bounding box.
  *  Rotated bounding box.
  *  Pixel-based segmentation.
* Paper Link: https://www.scitepress.org/Papers/2017/61206/61206.pdf 
* Dataset Link: https://www.kaggle.com/guofeng/hrsc2016 

### Ships in Satellite Imagery Dataset 
* The specific Dataset includes 4000 images labeled with either a "ship" or "no-ship" classification.
* These 4000 images are in .png format and have dimensions of 80 × 80 pixels. 
* The colour depth of the above 4000 images is 24 bit (8 bits per channel).
* Image chips were derived from PlanetScope full-frame visual scene products, which are orthorectified to a 3 meter pixel size.
* The "ship" class includes 1000 images. Images in this class are near-centered on the body of a single ship. Ships of different sizes, orientations, and atmospheric collection conditions are included.
* Each individual image filename follows a specific format: {label} __ {scene id} __ {longitude} _ {latitude}.png
  * label: Valued 1 or 0, representing the "ship" class and "no-ship" class, respectively.
  * scene id: The unique identifier of the PlanetScope visual scene the image chip was extracted from. 
  * longitude_latitude: The longitude and latitude coordinates of the image center point, with values separated by a single underscore.
* Dataset Link: https://www.kaggle.com/rhammell/ships-in-satellite-imagery 

### Airbus Ship Detection Challenge Dataset 
* The specific dataset consists of 192556 images with size of 768 × 768 pixels.
* These 192556 images contain  81723 ship instances.
* The above 192556 images have resolution of 96 dpi, and there are in .jpeg format.
* The colour depth of these images is 24-bit (8 bits per channel).
* For every one of these images there is a corresponding .csv file.
* The above .csv file contains the encoded pixel coordinates of each ship’s oriented rectangular mask, for every ship instance in the specific image.
* Dataset Link: https://www.kaggle.com/c/airbus-ship-detection/data 

### xView Dataset  
* The xView dataset contains 1 million objects across 60 classes in over 1,400 km2 of imagery and released by the Defense Innovation Unit Experimental (DIUx) and the National Geospatial-intelligence Agency (NGA).
* It consists of 1414 large scene images (GeoTIFF format) with dimensions ranging from 2772×2678 to 5121×3023 pixels.
* These data is collected from WorldView-3 satellites at 0.3 m ground sample distance.
* The GeoTIFF files are broken into two sets: RGB and 8-band. The 8-band set contains 8-band multispectral GeoTIFF images labeled as image_id.tif where image_id is a unique integer. Similarly, the RGB set contains pan-sharpened 3-band RGB GeoTIFF images with the same naming convention.
* The above 60 fine-grained classes are labeled in a parent class-child class manner. There are seven different parent classes, namely fixed-wing aircraft, passenger vehicle, truck, railway vehicle, maritime vessel, engineering vehicle, and building (some child classes have no parent class).
* The maritime vessel class contains nine child classes: motorboat; sailboat; tugboat; barge; fishing vessel; ferry; yacht; container ship; oil tanker.
* Each image set has a corresponding geoJSON file.The fields in the geoJSON files include: 
  * TYPE_ID : The bounding box label class ID.
  * CAT_ID : DigitalGlobe's unique ID for image strips.
  * IMAGE_ID : The image chip filename on which a feature is marked.
  * BOUNDS_IMCOORDS : Bounding box in pixel coordinates [xmin, ymin, xmax, ymax] of the image chip in which it is marked.
  * COORDINATES : Coordinates in longitude-latitude form for bounding box points.
* Paper Link: https://arxiv.org/abs/1802.07856 
* Dataset Link: https://challenge.xviewdataset.org/download-links 

### DOTA (Dataset for Object deTection in Aerial images) 
* It consists of 2806 images which they have dimensions of about 4000 × 4000 pixels. 
* This dataset contains 15 different categories but only 14 main categories( because small vehicle and large vehicle are both subcategories of vehicle).
* These 2806 images contain 43736 ship instances.
* Images in this dataset are collected from multiple sensors and platforms with multiple resolutions (Google Earth, Gaofen-2 and JL-1).
* In the dataset, each instance's location is annotated by a oriented quadrilateral bounding box. The first point of the bounding box (x1, y1) denotes the starting point, which refers to the top left corner of a ship.
* Annotations for an image are saved in a text file with the same file name. In the first line, 'imagesource'(from Google Earth, etc.) is given. In the second line, ’gsd’(ground sample distance) is given. From third line to the last line in the annotation text file, annotation for each instance is given.
* For every instance, a “difficult” label is provided, which indicates whether the instance is difficult to be detected (1 for difficult, 0 for not difficult).
* Paper Link: https://vision.cornell.edu/se3/wp-content/uploads/2018/03/2666.pdf 
* Dataset Link: https://captain-whu.github.io/DOTA/dataset.html 

### TGRS-HRRSD (High-Resolution Remote Sensing object Detection)
* It consists of 21761 images acquired from Google Earth and Baidu Map with the spatial resolution from 0.15m to 1.2 m.
* There are 13 object categories and 55740 object instances in HRRSD.
* These 13 categories are: ship, airplane, baseball diamond, basketball court, bridge, crossroad, ground track field, harbor, parking lot, storage tank, T-junction, tennis court and vehicle.
* The above 21761 images have various sizes and are in .jpeg format with 24-bit color depth.
* The HRRSD dataset contains 3975 ship instances.
* Every image has a corresponding .xml file which contains the image’s dimensions, the class name, and the 4 coordinates (of the lower left and upper right points) of each bounding box, for every instance that is shown in the image.
* Paper Link: - 
* Dataset Link: https://www.kaggle.com/haashaatif/tgrshrrsd-dataset 

### MASATI-v2 (MAritime SATellite Imagery dataset)
* The specific dataset consists of 7389 images with average dimensions of 512×512 pixels.
* The above 7389 images were obtained from Microsoft Bing maps and they are stored as .png format.
* In this dataset, each image has been manually labeled according to the following seven classes: 
    * Land : 1078 images 	| main class: Non-Ship 		| description: Land (no ships).
    * Coast : 1132 images 	| main class: Non-Ship 		| description: Coast (no ships).
    * Sea : 1022 images 	| main class: Non-Ship 		| description: Sea (no ships).
    * Ship : 1027 images 	          | main class: Ship 		| description: Sea with a ship (no coast).
    * Multi : 304 images 	          | main class: Ship 		| description: Multiple ships.
    * Coast-ship : 1037 images       | main class: Ship     	| description: Coast with ships.
    * Detail : 1789 images 	          | main class: Ship 		| description: Ship details. (Big Ship)
* This labeling is stored in XML files using the annotation format of PASCAL VOC.
* The above .xml files contain the image’s dimensions and channels, instance’s class name, and the four pairs of pixel coordinates for every instance’s  corresponding bounding box.
* Paper Link: https://www.mdpi.com/2072-4292/10/4/511 
* Dataset Link: https://www.iuii.ua.es/datasets/masati/ 

### DIOR(object Detection In Optical Remote sensing images)
* DIOR consists of 23463 optimal remote sensing images and 192472 object instances that are manually labeled with axis‐aligned bounding boxes, covered by 20 common object categories.
* The size of images in the dataset is 800×800 pixels and the spatial resolutions range from 0.5m to 30m.
* The above images acquired from Google Earth.
* The 2702 of the 23463 images contain ~63000 ship instances.  
* Paper Link: https://arxiv.org/abs/1909.00133 
* Dataset Link 1: https://pan.baidu.com/s/1Fc-zJtHy-6iIewvsKWPDnA 
* Dataset Link 2: http://www.escience.cn/people/gongcheng/DIOR.html

### FGSD (Fine-Grained Ship Detection)
* The specific dataset contains 2612 images from 17 large ports including China, Japan, the United States and Spain.
* The above images have dimensions of 930 × 930 pixels and their spatial resolutions range from 0.12m to 1.93m.
* These images contain 5634 ship instances.
* There are 43 categories of ships and a ’dock’ category in the dataset. 
* These 43 classes of ships were divided into 4 Level-2 categories, including warship, carrier, submarine and civil ship. And all ships shares the same level-1 label ship.
    * 1 Level-1 category : ship
    * 4 Level-2 categories : warship, carrier, submarine and civil ship
    * 43 Level-3 categories :  container, oil tanker, yacht, hovercraft, etc. 
* Ship samples in FGSD were annotated with both common used bounding box and rotated bounding box. The rotated bounding box is annotated as (xc, yc, w, h, θ).
* Each image has a corresponding annotation file, which includes the source-port’s ID, the resolution and corresponding Google Earth's resolution level of the image.
* Paper Link: https://arxiv.org/abs/2003.06832 
* Dataset: mail to :  ckyan@bupt.edu.cn

### PSDS (Peruvian Ship Data Set)
* It consists of 1310 images with size of 900 × 900 pixels.
* These 1310 images  include 9662 ship instances. 
* PSDS is created from 22 satellite images of PERUSAT-1 with 0.7 m spatial resolution.
* Every instance in the above images is labeled with a horizontal bounding box.
* Paper Link: https://iopscience.iop.org/article/10.1088/1742-6596/1642/1/012003 
* Dataset Link: -

### MSDS (Mini Ship Data Set)
* The specific dataset consists of generating 2993 images of 900 × 900 pixels.
* MSDS has been generated using Google Earth satellite images.
* The extracted 2993 images contain 4710 ship instances.
* Every instance in the above images is labeled with a horizontal bounding box.
* Paper Link: https://iopscience.iop.org/article/10.1088/1742-6596/1642/1/012003 
* Dataset Link: -

[Go To TOP](#TOP)