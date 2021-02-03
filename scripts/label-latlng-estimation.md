---
title: "Label Lat/Lng Estimation for Project Sidewalk"
author: "Mikey Saugstad"
date: "1/1/2021"
output:
  html_document: 
    toc: yes
    keep_md: yes
    df_print: kable
editor_options: 
  chunk_output_type: console
---

### Overview

We attempt to estimate the latitude and longitude of labels placed in Project Sidewalk using Google Street View (GSV). We use information like x and y pixels of the point clicked in the GSV image, the lat/lng of the camera, zoom level, etc. For ground truth, we use a large data set of lat/lng estimates using 3D LiDAR depth data that used to be provided by Google. The motivation for this analysis is that the API that used to provide this depth data was taken down by Google, and we need a semi-accurate alternative estimate to be used in Project Sidewalk. See Github issue [#2374](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/2374).

























### Methods

Determining latitude and longitude requires estimating the distance from the panorama and the heading angle (the regressions are predicting offset from the heading of the current view in GSV). Using those and the lat/lng of the camera that took the picture, we can estimate the lat/lng of the point that was clicked. We do this by finding traveling along the shortest path on an ellipsoid (using the `destPoint` function in the `geosphere` library). This will give very close results to the `destination` function in the `turf` library in JavaScript that we use in Project Sidewalk.

We used various estimation methods and compared the performance. We split the data into a training set with 80% of the labels (316,118 labels) and test set with 20% of the labels (79,029 labels). Here is a quick summary of each method of estimation:

1. Distance is 10 meters, heading difference is 0 (just use the camera's heading). Median error of 4.84 m.
1. Distance is determined by median distance in training set, heading difference is 0.  Median error of 4.64 m.
1. Distance is determined by median distance in training set _by label type_. Median error of 4.63 m.
1. Use a multivariate linear regression to predict heading angle and distance from the pano. We found the best performance using only `canvas_y` and `sv_image_y` as predictors. Median error of 3.42 m.
1. Use separate linear regressions for heading and distance instead of a multivariate. We found the best performance using `sv_image_y`, `canvas_y`, and `zoom` as predictors for distance, with `canvas_x` and `zoom` as predictors for heading. Median error of 1.79 m.
1. Use linear mixed effects models for both heading and distance. Similar to the linear regressions used in the previous estimate, but with `zoom` as a _random effect_ for both models, with the remaining variables as _fixed effects_. Median error of 1.79 m.
1. Use separate linear regressions for each zoom level. We found the best performance using `sv_image_y` and `canvas_y` as predictors for distance, and `canvas_x` as a predictor for heading. Median error of 1.47 m.

Here is a brief summary of the data cleaning. In total, we removed 97,152 labels (20%), ending up with 395,147 labels:

1. When creating linear regressions, we removed cases where the distance from the pano was greater than 50 meters so that the outliers didn't throw off results.
1. Removed labels from panos with 20 or more labels so that a few panos didn't have an outsized impact on the results.
1. Removed any labels with invalid lat/lng or canvas values.
1. Removed any labels that users marked as "deleted", since I sort of expect those to be less representative of a typical label. It shouldn't make a difference for the regressions, but for the estimates that use average distance, deleted labels could throw something off.
1. Removed any labels that did not have an estimate that used depth data, which happened if they were added after the removal of the depth data API endpoint.

### Results

The final verdict is that our final estimate (#7) was most accurate. This method involved making 6 linear regressions, 3 for heading and 3 for distance from the panorama. There are 3 linear regressions for each because we have a separate regression made for each of the 3 zoom levels. Using the coefficients from the regressions, distance and heading angle can be computed according to the formulas below.

The formula when zoom is 1:

* <code>distance from pano = 18.6051843 + 0.0138947 * sv_image_y + 0.0011023 * canvas_y</code>
* <code>difference in heading = -51.2401711 + 0.1443374 * canvas_x</code>

The formula when zoom is 2:

* <code>distance from pano = 20.8794248 + 0.0184087 * sv_image_y + 0.0022135 * canvas_y</code>
* <code>difference in heading = -27.5267447 + 0.0784357 * canvas_x</code>

The formula when zoom is 3:

* <code>distance from pano = 25.2472682 + 0.0264216 * sv_image_y + 0.0011071 * canvas_y</code>
* <code>difference in heading = -13.5675945 + 0.0396061 * canvas_x</code>


You can see that the biggest improvements came from moving from no regression to using some sort of regression (4.63 m to 3.42 m) and creating separate regressions for heading and distance from the pano (3.42 m to 1.79 m). And the final improvement of separating out regressions for each zoom level is sizable as well (1.79 m to 1.47 m)

Here are the results of each, all tables sorted by median error:

```r
summary_stats %>% relocate(estimate, median_error, sd_error)
```

<div class="kable-table">

|estimate   |median_error |sd_error |mean_error |min_error |max_error |
|:----------|:------------|:--------|:----------|:---------|:---------|
|error_est7 |1.47 m       |3.07 m   |2.39 m     |0.00 m    |34.76 m   |
|error_est5 |1.79 m       |3.15 m   |2.69 m     |0.01 m    |34.85 m   |
|error_est6 |1.79 m       |3.15 m   |2.69 m     |0.01 m    |34.85 m   |
|error_est4 |3.42 m       |3.44 m   |4.31 m     |0.01 m    |37.25 m   |
|error_est3 |4.63 m       |4.33 m   |5.47 m     |0.04 m    |42.32 m   |
|error_est2 |4.64 m       |4.37 m   |5.49 m     |0.01 m    |42.58 m   |
|error_est1 |4.84 m       |4.20 m   |5.56 m     |0.02 m    |42.17 m   |

</div>

```r
heading_error_stats
```

<div class="kable-table">

|estimate           |heading_error |
|:------------------|:-------------|
|heading_error_est7 |1.32°         |
|heading_error_est6 |3.19°         |
|heading_error_est5 |3.19°         |
|heading_error_est4 |14.29°        |
|heading_error_est1 |14.29°        |
|heading_error_est2 |14.29°        |
|heading_error_est3 |14.29°        |

</div>

```r
dist_error_stats
```

<div class="kable-table">

|estimate        |distance_error |
|:---------------|:--------------|
|dist_error_est7 |1.40 m         |
|dist_error_est4 |1.49 m         |
|dist_error_est5 |1.49 m         |
|dist_error_est6 |1.49 m         |
|dist_error_est3 |2.92 m         |
|dist_error_est2 |2.94 m         |
|dist_error_est1 |3.12 m         |

</div>
