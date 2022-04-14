import os
import argparse
import numpy as np
import cv2
# import readscan
# from readscan import *

def readimg(fpath, ch='0', varargin=None):
    imout = None

    if isinstance(fpath, list):
        imout = subprocess(fpath, ch, len(fpath))
        return imout

    elif not os.path.exists(fpath):
        print('not a valid file path')
        return imout
    
    base_dir_pair = os.path.split(fpath)
    parentdr = base_dir_pair[0]
    split_tup = os.path.splitext(base_dir_pair[1])
    print(parentdr, split_tup[0], split_tup[1])

    if split_tup[1]=='.png':
        imout = cv2.imread(fpath, 0)

    elif split_tup[1]=='.yaml':
        sdata = readscan(fpath)
        img_num = len(sdata.images)
        imout = subprocess(sdata.images, ch, img_num)
    return imout

def subprocess(path_list, ch, img_num):
    channels = [0]
    if isinstance(ch, str):
        if ch=='all':
            channels = [i for i in range(img_num)]
        elif ch>='1' and ch<=str(img_num):
            channels = [int(ch)-1]
    elif isinstance(ch, int):
        min_num = min(ch, img_num)
        max_num = max(min_num, 1)
        channels = [max_num-1]

    imout = None
    i = 0
    for ch_idx in channels:
        if not os.path.isfile(path_list[ch_idx]) or not path_list[ch_idx].lower().endswith('.png'):
            continue
        print(i, path_list[ch_idx])
        img = cv2.imread(path_list[ch_idx], 0)
        if i==0:
            imout = img
        else:
            imout = np.dstack((imout, img))    
        i += 1
    return imout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', action='store', help='Input a file')
    arguments = parser.parse_args()


    imout = readimg(arguments.inputfile, ch='all')
    if imout is not None:
        im_shape = imout.shape
        if len(im_shape) == 2:
            cv2.imwrite('result.png', imout)
        else:
            cv2.imwrite('result.png', imout[:,:,0])
