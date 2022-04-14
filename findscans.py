import os
import numpy as np
import struct
import argparse
import cv2

import readscan
from readscan import *
import readimg
from readimg import *

class Scan():
    def __init__(self):
        self.yamlpath = None
        self.tmdpath = None
        self.nrmpath = None
        self.tmdfiles = None
        self.nrmfiles = None
        self.thumbnail = None

    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


def findscans(spath, scannm=[], tag=[]):
    scans = []
    if (os.path.isdir(spath)):
        for dname, dirs, files in os.walk(spath):
            if 'scan.yaml' in files and len(scannm)==0:
                s = Scan()
                s.name = dname
                s.yamlpath = os.path.join(dname, 'scan.yaml')
                s.thumbnail = os.path.join(dname, 'thumbnail.jpg')
                s.tmdpath = ''
                s.nrmpath = ''
                s.tmdfiles = []
                s.nrmfiles = []

                scans.append(s)
    
    for i, scan in enumerate(scans):
        if scan.tmdpath == '':
            scans[i] = add3dfiles(scan, tag)
    
    return scans

def add3dfiles(scan, tag):
    tmdfiles = []
    nrmfiles = []
    tmdpath = ''
    nrmpath = ''
    base_dir_pair = os.path.split(scan.yamlpath)
    parentdr = base_dir_pair[0]
    for dname, dirs, files in os.walk(parentdr):
        for fname in files:
            if fname.lower().endswith('.tmd'):
                tmdfiles.append(os.path.join(dname, fname))

            # if fname.lower().endswith('_nrm.png'):
            #     nrmfiles.append(os.path.join(dname, fname))

    if len(tmdfiles) and len(tmdpath) == 0:
        tmdpath = tmdfiles[0]

    # if len(nrmfiles) and len(nrmpath) == 0:
    #     nrmpath = nrmfiles[0]

    #  If the tag isn't empty, make sure we found either tmdpath or nrmpath
    if len(tag) and len(nrmpath)==0 and len(tmdpath)==0:
        print('no TMD or normal map found with tag = %s',tag)

    # Check for normal maps with same names as TMD files
    if len(tmdpath):
        base_dir_pair_tmd = os.path.split(tmdpath)
        tmdparent = base_dir_pair_tmd[0]
        split_tmd = os.path.splitext(base_dir_pair_tmd[1])
        tmdname = split_tmd[0]
        tmdext = split_tmd[1]
        nrmpath = os.path.join(tmdparent, tmdname+'_nrm.png')

        if not os.path.exists(nrmpath):
            nrmpath = ''
            
    for tmd_path in tmdfiles:
        base_dir_pair_tmd_i = os.path.split(tmd_path)
        tmdparent_i = base_dir_pair_tmd_i[0]
        split_tmd_i = os.path.splitext(base_dir_pair_tmd_i[1])
        tmdname_i = split_tmd_i[0]
        tmdext_i = split_tmd_i[1]
        localnrmpath = os.path.join(tmdparent_i,tmdname_i+'_nrm.png')
        if not os.path.exists(localnrmpath):
            localnrmpath = ''
        nrmfiles.append(localnrmpath)

        
    scan.tmdpath  = tmdpath
    scan.nrmpath  = nrmpath
    scan.tmdfiles = tmdfiles
    scan.nrmfiles = nrmfiles

    return scan

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', action='store', help='Input dir')
    arguments = parser.parse_args()
    # e.g. python findscans.py "..\..\..\Public\Documents\GelSight\Scans\Essex_Furukawa"
    scans = findscans(arguments.inputfile)
    for scan in scans:
        sdata = readscan(scan.yamlpath)
        # imout = readimg(sdata.images, ch='all')
        for ann in sdata.annotations:
            print(ann)
