# Introduction to gspython stitch package

The stitch package contains Python tools to process and stitch together photographs and heightmap data collected from GelSight scans. The module `stitchscan` contains all of the methods required. For convenience, the package also includes a command line script `stitch.py` as well as an interactive notebook `test-stitch.ipynb` which walks through the structure of the module.

## Requirements

open source module `numpy`

open source module `scipy`

open source module `opencv`

open source module `matplotlib`

open source package `gspython` with modules `readscan` and `readtmd`

## GelSight output file directory structure

When using the GelSight system to make scans of a single object, the system outputs a number of folders named `Scan001`, `Scan002`, and so on, which are numbered in the order that the scan is performed. Each of these folders contains six PNG images which correspond to photographs taken with illumination from different angles, a composite photograph named `thumbnail.jpg`, a raw `Scan***.tmd` file which contains heightmap information, and a raw `scan.yaml` file with additional scan data. **In order to use the stitchscan package, you must make scans of the object such that each pair of consecutive scans includes  an overlapping window, then collect each of the `Scan***` folders corresponding to a single object into its own directory `object_name/`, without changing their internal structure.** 

1. Object1/
    * Scan001/
        * image01.png
        * image02.png
        * image03.png
        * image04.png
        * image05.png
        * image06.png
        * scan.yaml
        * Scan001.tmd
        * thumbnail.jpg
        
        ...
    * Scan002/
        
        ...
    
    ...
    * Scan***/
        
        ...
2. Object2/
    
    ...

...

## TODO
1. Additional warnings for bad matches
2. Linear blending for seams on heightmaps