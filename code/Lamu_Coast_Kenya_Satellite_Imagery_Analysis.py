from glob import glob

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import plotly.graph_objects as go

np.seterr(divide='ignore', invalid='ignore')

#DATA SETS
#Lamu coast data

S_sentinel_bands = glob("https://drive.google.com/drive/folders/1ghhTqjdmWu9WoABaPlf77s1gmD2lW81p?usp=sharing*B?*.tif")
S_sentinel_bands.sort()
S_sentinel_bands

l = []
for i in S_sentinel_bands:
  with rio.open(i, 'r') as f:
    l.append(f.read(1))
    arr_st = np.stack(l)
    arr_st.shape

#RGB Composite image    
ep.plot_bands(arr_st, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()

rgb = ep.plot_rgb(arr_st, 
                  rgb=(3,2,1), 
                  figsize=(10, 16), 
                  # title='RGB Composite Image'
                  )

plt.show()

ep.plot_rgb(
    arr_st,
    rgb=(3, 2, 1),
    stretch=True,
    str_clip=0.2,
    figsize=(10, 16),
    # title="RGB Composite Image with Stretch Applied",
)

plt.show()

colors = ['tomato', 'navy', 'MediumSpringGreen', 'lightblue', 'orange', 'blue',
          'maroon', 'purple', 'yellow', 'olive', 'brown', 'cyan']

ep.hist(arr_st, 
         colors = colors,
        title=[f'Band-{i}' for i in range(1, 13)], 
        cols=3, 
        alpha=0.5, 
        figsize = (12, 10)
        )

plt.show()

#VEGETATION AND SOIL INDICES
#Normalized Difference Vegetation index (NDVI)

#NDVI = ((NIR - Red)/(NIR + Red))
#where: NIR is Pixel values from the near-infrared band
#       Red is Pixel values from the red band

ndvi = es.normalized_diff(arr_st[7], arr_st[3])

ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()

#Soil-Adjusted Vegetation index  (SAVI)
#SAVI = ((NIR - Red)/(NIR + Red + L)) * (1 + L)
#where: L is the amount of green vegetation cover

L = 0.5

savi = ((arr_st[7] - arr_st[3]) / (arr_st[7] + arr_st[3] + L)) * (1 + L)

ep.plot_bands(savi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()

#Visible Atmospherically Resistant Index (VARI)
#VARI = (Green - Red)/(Green + Red - Blue)
#where: Green is Pixel values from the green band
#       Blue  is pixel values from the blue band

vari = (arr_st[2] - arr_st[3])/ (arr_st[2] + arr_st[3] - arr_st[1])
ep.plot_bands(vari, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()

#Distribution of NVDI,SAVI and VARI Pixel values
ep.hist(np.stack([ndvi, savi, vari]), 
        alpha=0.5,
        cols=3, 
        figsize=(20, 5),
        title = ['NDVI', 'SAVI', 'VARI'],
        colors = ['mediumspringgreen', 'tomato', 'navy'])
plt.show()

#Water Indices
#Modified Normalized Difference Water Index (MNDWI)
#MNDWI = (Green - SWIR)/(Green + SWIR)
#where: SWIR is pixel value from the short-wave infrared band

mndwi = es.normalized_diff(arr_st[2], arr_st[10])
ep.plot_bands(mndwi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()

#Normalized Difference Moisture Index(NDMI)
#NDMI = (NIR - SWIR1)/(NIR + SWIR1)
#where: SWIR1 is pixel value from the short-wave infrared 1 band

ndmi = es.normalized_diff(arr_st[7], arr_st[10])
ep.plot_bands(ndmi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


#Geology Indices
#Clay Minerals
#Clay Minerals Ratio = SWIR1 / SWIR2
#where: SWIR2 is pixel values from the short-wave onfrared 2 band

cmr = np.divide(arr_st[10], arr_st[11])
ep.plot_bands(cmr, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()

#Ferrous Minerals
#Ferrous Minerals Ratio = SWIR / NIR

fmr = np.divide(arr_st[10], arr_st[7])
ep.plot_bands(fmr, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()
