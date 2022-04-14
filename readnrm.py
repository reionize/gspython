import os
import numpy as np
import struct
import argparse
import cv2

def readnrm(fpath):
#   The return value NRM is an M-by-N-by-3 array containing the unit-length
#   surface normal at every pixel, with the X, Y and Z components of the 
#   surface normal saved in the channels 1-3, respectively. The values
#   of the X and Y components span the range -1 to 1 and the values of the 
#   Z component span the range 0-1. Positive Z values of the surface normal
#   face the observer.

    if not os.path.isfile(fpath):
        print('not a valid file path')
        return None
    nrm = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    nrm = 2*nrm - 1
    return nrm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', action='store', help='Input a nrm file')
    arguments = parser.parse_args()
    
    nrm = readnrm(arguments.inputfile)