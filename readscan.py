import os
import yaml
import argparse

class SData():
    def __init__(self):
        self.version = None
        self.calib = None
        self.crop = None
        self.guid = None
        self.mmperpixel = None
        self.images = None
        self.annotations = None

    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


class Annotation():
    def __init__(self):
        self.type = None
        self.name = None
        self.label = None
        self.id = None
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.x = None
        self.y = None
        self.gx = None
        self.gy = None
        self.r = None
        self.w = None
        self.h = None
        self.closed = None
        self.points = None

    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


def readscan(fpath):
    ## fpath is the scan dir
    if os.path.isdir(fpath):
        fpath = os.path.join(fpath, 'scan.yaml')
    ## fpath is the scan.yaml file
    if (not os.path.isfile(fpath)) or (not fpath.lower().endswith('.yaml')):
        print('not a valid yaml path')
        return None
    
    base_dir_pair = os.path.split(fpath)
    parentdr = base_dir_pair[0]

    sdata = SData()
    with open(fpath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        sdata.version = data_loaded['version']
        sdata.calib = findcalib(data_loaded['calib'], parentdr)
        sdata.crop = data_loaded['crop']
        sdata.guid = data_loaded['guid']
        sdata.mmperpixel = data_loaded['mmperpixel']
        sdata.images = loadimages(data_loaded['images'], parentdr)
    
    if sdata.version >= 2:
        scancontext = os.path.join(parentdr, 'Analysis/scancontext.yaml')
        if not os.path.isfile(scancontext):
            print('not a valid file path')
        else:
            sdata.annotations = loadshapesasannotations(scancontext)

    return sdata

def loadimages(filelist, parentdr):
    impaths = []
    for name in filelist:
        name = os.path.join(parentdr, name)
        impaths.append(name)
    return impaths

def loadshapesasannotations(fpath):
    with open(fpath, 'r') as stream:
        fd = yaml.safe_load(stream)
        annotations = None
        if fd.get('shapes'):
            annotations = loadannotations(fd['shapes'])

        return annotations

def loadannotations(fd):
    annotations = []
    for dict_i in fd:
        annotation = Annotation()
        annotation.type = dict_i.get('type')
        annotation.name = dict_i.get('name')
        annotation.label = dict_i.get('label')
        annotation.id = dict_i.get('id')
        annotation.x1 = dict_i.get('x1')
        annotation.x2 = dict_i.get('x2')
        annotation.y1 = dict_i.get('y1')
        annotation.x = dict_i.get('x')
        annotation.y = dict_i.get('y')
        annotation.gx = dict_i.get('gx')
        annotation.gy = dict_i.get('gy')
        annotation.r = dict_i.get('r')
        annotation.w = dict_i.get('w')
        annotation.h = dict_i.get('h')
        annotation.closed = dict_i.get('closed')
        annotation.points = parsePointList(dict_i.get('points'))
        annotations.append(annotation)
    return annotations

def parsePointList(point_list):
    if point_list is None:
        return None
    coordinates = []
    points = []
    for item in point_list:
        item = item.replace('(', '')
        item = item.replace(')', '')
        item = float(item)
        coordinates.append(item)
    matches = int(len(coordinates)/2)
    for i in range(matches):
        p = (coordinates[2*i], coordinates[2*i+1])
        points.append(p)

    return points


def findcalib(calib_file, parentdr):
    if calib_file == '':
        return None
    calib_path = os.path.join(parentdr, calib_file)
    if os.path.exists(calib_path):
        return calib_path
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', action='store', help='Input a yaml file')
    arguments = parser.parse_args()
    sdata = readscan(arguments.inputfile)
    print(sdata)