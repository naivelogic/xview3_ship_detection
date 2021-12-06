# %%
import pandas as pd
import json

from pandas.io.json import json_normalize
from pathlib import Path
import numpy as np

# %% [markdown]
# ### load detection json/csv file

# %%
# Getting predictions from file - CSV
inference_csv = '/mnt/omreast_users/phhale/open_ds/xview3/experiments/yolact/112921_yolact_r50_xView3_npy200_v1_bg/50k_590dd08f71056cacv.csv'
df_preds = pd.read_csv(inference_csv)
# Renaming columns for convenience
df_preds.rename({'detect_scene_row':'scene_rows', 'detect_scene_column':'scene_cols'}, inplace=True, axis='columns')

# %% [markdown]
# # Visualize predictions

# %%
import matplotlib as mpl
from matplotlib import pyplot as plt
import rasterio

vmin, vmax = -35, -5

def display_image_in_actual_size(im_data, rows, cols, rows2=None, cols2=None):

    dpi = mpl.rcParams['figure.dpi']
    height, width= im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    print('Plotting image...')
    ax.matshow(im_data, cmap="bone", vmin=vmin, vmax=vmax)
    
    print('Plotting scatterplot...')
    ax.scatter(cols, rows, s=120, facecolors="none", edgecolors="r")
    if rows2 is not None:
        #ax.scatter(cols2, rows2, s=80, facecolors="none", edgecolors="g")
        ax.scatter(cols2, rows2, s=280, facecolors="none", edgecolors="g")
    
    plt.margins(0,0)
    plt.show()

# %%
# Scene ID
#scene_id = '590dd08f71056cacv'
val_label_file_csv = '/mnt/omreast_users/phhale/open_ds/xview3/datasets/labels/validation.csv'
scene_id = df_preds['scene_id'][0] #DEBUG_SCENE_ID #"0157baf3866b2cf9v" #args.scene_ids = "0157baf3866b2cf9v"

# Getting detect df
df = pd.DataFrame()
detects = pd.read_csv(val_label_file_csv) #val_label_file
df = pd.concat((df, detects))
df.rename({'detect_scene_row':'scene_rows', 'detect_scene_column':'scene_cols'}, inplace=True, axis='columns')

df = df[df['scene_id'] == scene_id]

score_all = False
#score_all = True
# By default we only score on high and medium confidence labels
if not score_all:
    df = df[df['confidence'].isin(['HIGH', 'MEDIUM'])].reset_index()


# %%
# Loading image in UTM coordinates
#image_folder = "/mnt/omreast_users/phhale/open_ds/xview3/datasets/dev/one_sample/validation"
image_folder = '/mnt/mnt_xview3/datasets/validation'
data_root = Path(image_folder) #Path(args.image_folder) #"/mnt/omreast_users/phhale/open_ds/xview3/datasets/dev/one_sample/validation" #val_data_root
grdfile = data_root / scene_id / 'VH_dB.tif'#'bathymetry.tif' #'VH_dB.tif' #'VV_dB.tif'
src = rasterio.open(grdfile)
image_orig = src.read(1)


# %%
scene_id

# %%
# Identifying a specific detection on which to center plot
#row = df[df['detect_id'].str.contains('6.46312')].iloc[0]
#row = df[df['detect_id'].str.contains('6.2')].iloc[0]
#row = df[df['detect_id'].str.contains('6.5')].iloc[0]
#row = df[df['detect_id'].str.contains('6.4')].iloc[0]
#row = df[df['detect_id'].str.contains('5.15578')].iloc[0]
#row = df[df['detect_id'].str.contains('5.661389')].iloc[0]
row = df[df['detect_id'].str.contains('6.3')].iloc[0]
#print(row)

# %%
# Defining size of image patch for plot (generally suggest keeping ~ 1000)
patch_half_width = 500 #1000 #750

# Getting predictions and detections in image patch 
df_small = df[np.abs(df.scene_rows - row.scene_rows) < patch_half_width]
df_small = df_small[np.abs(df_small.scene_cols - row.scene_cols) < patch_half_width]

df_preds_small = df_preds[np.abs(df_preds.scene_rows - row.scene_rows) < patch_half_width]
df_preds_small = df_preds_small[np.abs(df_preds_small.scene_cols - row.scene_cols) < patch_half_width]

dt = image_orig[row.scene_rows-(patch_half_width):row.scene_rows+(patch_half_width), row.scene_cols-(patch_half_width):row.scene_cols+(patch_half_width)]

# Plotting ground truth (red) and predicted (green) detections
display_image_in_actual_size(dt, 
                            df_small.scene_rows - row.scene_rows + patch_half_width,
                            df_small.scene_cols - row.scene_cols + patch_half_width,
                            rows2=df_preds_small.scene_rows - row.scene_rows + patch_half_width,
                            cols2=df_preds_small.scene_cols - row.scene_cols + patch_half_width,
                            )

# %%


# %%
display_image_in_actual_size(dt, 
                            df_small.scene_rows - row.scene_rows + patch_half_width,
                            df_small.scene_cols - row.scene_cols + patch_half_width, 
                            rows2=None,cols2=None)

# %%


# %%



