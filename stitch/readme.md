# Introduction to gspython stitch package

The stitch package contains Python tools to process and stitch together photographs and heightmap data collected from GelSight scans. The module `stitchscan` contains all of the methods required. For convenience, the package also includes a command line script `stitch.py` as well as an interactive notebook `test-stitch.ipynb` which walks through the structure of the module.

## Requirements

Python 3.x

open source module `numpy`

open source module `scipy`

open source module `opencv`

open source module `matplotlib`

open source package `gspython` with modules `readscan` and `readtmd`

## GelSight output file directory structure

When using the GelSight system to make scans of a single object, the system outputs a number of folders named `Scan001`, `Scan002`, and so on, which are numbered in the order that the scan is performed. Each of these folders contains six PNG images which correspond to photographs taken with illumination from different angles, a composite photograph named `thumbnail.jpg`, a raw `Scan***.tmd` file which contains heightmap information, and a raw `scan.yaml` file with additional scan data. 

In order to fit together scans with the stitchscan package, you must make scans of each object such that every pair of consecutive scans includes  an overlapping window, then collect each of the `Scan***` folders corresponding to a single object into its own directory `object_name/`, without changing their internal structure, as shown below. 

Currently, the `stitchscan` module and `stitch.py` script only support stitching together a single long strip. However, it is possible to circumvent this limitation by first stitching together vertical strips, then rotating all of the strips before stitching them together horizontally. 

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
2. Object2/
    
    ...

etc.

## Flowchart description of stitch.py routine

To use the command line script, make sure that Python 3.x is enabled and type:

<div>
    
python package_directory/stitch.py data_directory/Object1 data_directory/Object2 ... data_directory/Object*
    
</div>

You can use `python package_directory/stitch.py --help` or `python package_directory/stitch.py -h` to view additional options. An overview of the stitching routine is provided below. By default, the script writes all results to the path `package_directory/output/`. 

<p align="center"> <b>
stitch.py main() script
</b> </p>

$$ \downarrow $$

<p align="center"> <b>
    stitchscan.stitchscans() call 
</b> </p>

$$ \downarrow $$

<p align="center"> <i>
    Loop through each object
    
    stitchscan.readdata() call to get scan data for object
    
    Iterate through pairs of consecutive scans
</i> </p>

$$ \downarrow $$

<p align="center">
    stitchscan.getmatches() call to match features in pair of photographs
    
    Compute 2D histogram of matches within scan
    
    stitchscan.denserect() call to select area with good signal-to-noise ratio
    
    Compute least-squares affine transformation from matches in selected area
    
    Blend overlapping region to obtain stitched image
</p>

$$ \downarrow $$

<p align="center"> <i>
    Output final stitched photograph and heightmap of entire object
</i> </p>

## TODO
1. Specify output directory
2. Additional warnings for bad matches
3. Linear blending for seams on heightmaps