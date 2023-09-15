# Satellite_project
 ##  Satellite Imagery Analysis

This repository contains code and data for analyzing satellite imagery of the Lamu Coast in Kenya. The goal of this analysis is to identify different land cover types and water bodies in the region.

### Data

The data used in this analysis is from the Sentinel-2 satellite. The data consists of 12 bands, each representing a different wavelength of light. The bands are:

* Band 1: Coastal aerosol
* Band 2: Blue
* Band 3: Green
* Band 4: Red
* Band 5: Vegetation red edge
* Band 6: Vegetation red edge
* Band 7: Near infrared
* Band 8: Near infrared
* Band 9: Water vapor
* Band 10: Short-wave infrared
* Band 11: Short-wave infrared
* Band 12: Short-wave infrared

### Preprocessing

The first step in the analysis is to preprocess the data. This involves:

* Resampling the data to a common resolution
* Converting the data to a format that can be used by the machine learning algorithms
* Normalizing the data

### Machine Learning Algorithms

The following machine learning algorithms are used to analyze the data:

* K-Means clustering
* Support Vector Machine (SVM)
* Random Forest
* Light GBM

### Results

The results of the analysis show that the K-Means clustering algorithm is the best at classifying the land cover types and water bodies in the region. The SVM and Random Forest algorithms also perform well, but the Light GBM algorithm does not perform as well.

### Conclusion

The results of this analysis can be used to inform decision-making about land use and management in the Lamu Coast region. The data can also be used to monitor changes in the environment over time.

### Code

The code used to perform the analysis is provided in the `code` directory. The code is organized into the following files:

* `Classification/Land_Cover_classification.py`: This file contains the code for classifying the land cover types in the region.
* `Clustering/Naivasha_ground_truth_labelling_of_satellite_imagery_using_K-means_Clustering.py`: This file contains the code for clustering the data into different groups.
* `code/Lamu_Coast_Kenya_Satellite_Imagery_Analysis.py`: This file contains the code

