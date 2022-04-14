import os
import numpy as np
import struct
import argparse
import cv2

class Header():
    def __init__(self):
        self.imW = None
        self.imH = None
        self.lengthx = None
        self.lengthy = None
        self.offsetx = None
        self.offsety = None
        self.mmpp = None
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


def readtmd(fpath):
    heightMap = None
    headerData = None
    if not os.path.exists(fpath) or not fpath.lower().endswith('.tmd'):
        return heightMap, headerData

    # print('**********')
    header_len = 32
    comment_len = 2048
    int32_len = 4
    float_len = 4

    with open(fpath, 'rb') as f:
        tmd = f.read()

        headerData = Header()

        start = 0
        end = start+header_len
        TMD_HEADER = tmd[start:end].decode('UTF-8')
        # print(TMD_HEADER)
        
        start = end
        end += comment_len
        # commentbuffer = tmd[start:end]
        # print(tmd[start:end+100])

        while start<end:
            if tmd[start]==0:
                end = start+1
                break
            start += 1

        start = end
        end += int32_len
        imW_byte = tmd[start:end]
        headerData.imW = struct.unpack('i', imW_byte)[0]

        start = end
        end += int32_len
        imH_byte = tmd[start:end]
        headerData.imH = struct.unpack('i', imH_byte)[0]

        start = end
        end += float_len
        lengthx_byte = tmd[start:end]
        headerData.lengthx = struct.unpack('f', lengthx_byte)[0]

        start = end
        end += float_len
        lengthy_byte = tmd[start:end]
        headerData.lengthy = struct.unpack('f', lengthy_byte)[0]

        start = end
        end += float_len
        offsetx_byte = tmd[start:end]
        headerData.offsetx = struct.unpack('f', offsetx_byte)[0]

        start = end
        end += float_len
        offsety_byte = tmd[start:end]
        headerData.offsety = struct.unpack('f', offsety_byte)[0]

        headerData.mmpp = headerData.lengthx/headerData.imW

        pxOffX = int(headerData.offsetx/headerData.mmpp)
        pxOffY = int(headerData.offsety/headerData.mmpp)
        fullW = headerData.imW + pxOffX
        fullH = headerData.imH + pxOffY

        heightMap = np.zeros((int(fullH), int(fullW)), dtype=np.float32)
        # heightMap = np.zeros((headerData.imH, headerData.imW), dtype=np.float32)
        for y in range(headerData.imH):
            start = end
            end += float_len*headerData.imW
            heightMap[y+pxOffY:y+pxOffY+1][pxOffX:]= struct.unpack('f'*headerData.imW, tmd[start:end])

    return heightMap, headerData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', action='store', help='Input a yaml file')
    arguments = parser.parse_args()
    
    hm, hdata = readtmd(arguments.inputfile)
    print(hdata)
    hm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('heightmap.png', hm)    