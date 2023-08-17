import os
import sys
import heapq
import numpy as np
import scipy
import random
import cv2 as cv
from matplotlib import pyplot as plt

sys.path.append(r"../")
import readscan
import readtmd

def readdata(parent,allowance=10,baseline=75,normalize_height=True):
    """
    Crawls parent directory to find scans of object and reads in heightmap and photographic data, optionally normalizing for background variation. Returns list of scan folders, full heightmap information, list of heightmaps which are (optionally) normalized with the same allowance region, and list of corresponding images.
    
    Parameters
    ----------
    parent : str
        Parent directory in which scans of object are located.
    allowance : int, optional
        Width parameter in pixels which determines size of corner regions to sample from when subtracting out bilinear gradient to minimize variation in background brightness. Default is 10.
    baseline: int, optional
        Baseline brightness parameter between 0-255 which is added to entire photograph. Default is 75.
    normalize_height: bool, optional
        Option to normalize heightmap by sampling same corner regions and subtracting out the resulting bilinear gradient.
    
    Returns
    -------
    scans : list
        List of scan directories
    hdata : dict of tuples
        Dictionary storing as values heightmap data in the form of a 2-tuple containing the heightmap and additional scan information, which are mapped to by keys given by scan position. 
    hmaps : list of numpy.ndarrays
        List of ndarrays containing height data at every pixel location in each scan, represented from 0-1. 
    ims : list of numpy.ndarrays
        List of ndarrays containing photographic image for each scan, with each pixel represented in RGB format as a list of ints from 0-255. Image is expected to be grayscale, so each pixel is expected to be of the form [value, value, value].
    """
    
    scans = []
    hdata = {}
    hmaps = []
    ims = []
    
    # crawl directory and find individual scans
    for root, dirs, files in os.walk(parent):
        for name in dirs:
            if 'Scan' in name:
                heapq.heappush(scans,name)
    count = len(scans)
    sorts = []
    for i in range(count):
        sorts.append(heapq.heappop(scans))
    scans = sorts

    # for each scan read in heightmap and corresponding composite photograph
    for i in range(len(scans)):
        s = scans[i]
        spath = parent + s + '/' + 'scan.yaml'
        if not os.path.exists(spath):
            print('Cannot find raw scan file for specified path')
            break
    
        hpath = parent + s + '/' + s + '.tmd'
        if not os.path.exists(hpath):
            print('Cannot find TMD file for specified path')
            break

        hdata[i] = readtmd.readtmd(hpath)
        im = cv.imread(parent + s + '/' + 'thumbnail.jpg')
        
        # normalize corners of image by subtracting out mask      
        height,width,a = im.shape
        corners = [np.average(im[:allowance, :allowance], axis = (0,1)),np.average(im[:allowance, -allowance:], axis = (0,1)),np.average(im[-allowance:, -allowance:], axis = (0,1)),np.average(im[-allowance:, :allowance], axis = (0,1))]
        gradient = np.floor([[corners[0],corners[1]],[corners[3],corners[2]]])-[[[baseline]*3,[baseline]*3],[[baseline]*3,[baseline]*3]]
        mask = cv.resize(gradient,None,fx=width/2,fy=height/2,interpolation=cv.INTER_LINEAR)
        
        im = im-mask
        im = im.clip(0,255).astype(int)
        im = cv.normalize(im, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        ims.append(im)

        # normalize gradient for heightmap if option is turned on        
        hm = hdata[i][0]
        if normalize_height:
            corners = [np.average(hm[:allowance, :allowance], axis = (0,1)),np.average(hm[:allowance, -allowance:], axis = (0,1)),np.average(hm[-allowance:, -allowance:], axis = (0,1)),np.average(hm[-allowance:, :allowance], axis = (0,1))]
            gradient = np.floor([[corners[0]*255,corners[1]*255],[corners[3]*255,corners[2]*255]])
            mask = cv.resize(gradient,None,fx=width/2,fy=height/2,interpolation=cv.INTER_LINEAR)
            hm = hm-mask/255
        hmaps.append(hm)
        
    return scans,hdata,hmaps,ims

def denserect(nbins, bins):
    """
    Given 2D histogram data, returns rectangle bounding four largest bins. 
    
    Parameters
    ----------
    nbins: int
        Number of bins in each axis. 
    bins: array_like
        2D histogram data with same number of bins in each axis. 
    
    Returns
    -------
    lims : list of ints
        List of the form [y1, y2, x1, x2] with y1 <= y2 and x1 <= x2 defining the bounding rectangle.
    """
    
    rect = {1:(0,(0,0)),2:(0,(0,0)),3:(0,(0,0)),4:(0,(0,0))}
    for r in range(nbins):
        for c in range(nbins):
            if bins[r][c] >= rect[1][0]:
                rect[4] = rect[3]
                rect[3] = rect[2]
                rect[2] = rect[1]
                rect[1] = (bins[r][c], (r,c))
            elif bins[r][c] >= rect[2][0]:
                rect[4] = rect[3]
                rect[3] = rect[2]
                rect[2] = (bins[r][c], (r,c))
            elif bins[r][c] >= rect[3][0]:
                rect[4] = rect[3]
                rect[3] = (bins[r][c], (r,c))
            elif bins[r][c] >= rect[4][0]:
                rect[4] = (bins[r][c], (r,c))
    lims = [rect[1][1][0],rect[1][1][0],rect[1][1][1],rect[1][1][1]]
    for i in [2,3,4]:
        if rect[i][1][0] < lims[0]:
            lims[0] = rect[i][1][0]
        elif rect[i][1][0] > lims[1]:
            lims[1] = rect[i][1][0]
        if rect[i][1][1] < lims[2]:
            lims[2] = rect[i][1][1]
        elif rect[i][1][1] > lims[3]:
            lims[3] = rect[i][1][1]
    return lims

def stitchborderpyr(im1, im2, levels=5):
    """
    Uses Gaussian pyramids to vertically blend two images. Based on OpenCV demo code at https://docs.opencv.org/3.4/dc/dff/tutorial_py_pyramids.html.
    
    Parameters
    ----------
    im1: numpy.ndarray
        Image which will be stacked on the bottom, in a standard OpenCV format.
    im2: numpy.ndarray
        Image which will be stacked on the top, in the same format as im1.
    levels: int, optional
        Number of levels of pyramids to use.
    
    Returns
    -------
    result : numpy.ndarray
        Vertically blending image with im2 on top of im1, in the same format as the inputs.
    """
    
    # generate Gaussian pyramids
    G = im1.copy()
    gp1 = [G]
    for i in range(levels):
        G = cv.pyrDown(G)
        gp1.append(G)
        
    G = im2.copy()
    gp2 = [G]
    for i in range(levels):
        G = cv.pyrDown(G)
        gp2.append(G)
        
    # generate Laplacian pyramids
    lp1 = [gp1[levels-1]]
    for i in range(levels-1,0,-1):
        b = gp1[i-1].shape
        GE = cv.pyrUp(gp1[i],dstsize=(b[1],b[0]))
        L = cv.subtract(gp1[i-1],GE)
        lp1.append(L)

    lp2 = [gp2[levels-1]]
    for i in range(levels-1,0,-1):
        b = gp2[i-1].shape
        GE = cv.pyrUp(gp2[i],dstsize=(b[1],b[0]))
        L = cv.subtract(gp2[i-1],GE)
        lp2.append(L)
        
    # Add top and bottom halves of images in each level
    LS = []
    for l1,l2 in zip(lp1,lp2):
        rows,cols,dpt = l1.shape
        ls = np.vstack((l2[0:rows//2], l1[rows//2:]))
        LS.append(ls)
    result = LS[0]
    for i in range(1,levels):
        b = LS[i].shape
        result = cv.pyrUp(result,dstsize=(b[1],b[0]))
        result = cv.add(result, LS[i])
        
    return result

def stitchscans(parentdir, fdata, allowance=20, normalize_height=True, nbins=10, ratio=0.7, levels=0, diagnose=False, mode='sift', tag=''):
    """
    Reads and stitches together scans of objects in fdata. Note that when using diagnostic mode or stitching multiple images in a script, stitching process for each image will not continue until current diagnostic image is closed, as cv.waitKey() is not specified.
    
    Parameters
    ----------
    parentdir: type, optional?
        lorem ipsum dolor
    fdata: type, optional?
        lorem ipsum dolor
    allowance: type, optional
        lorem ipsum dolor
    normalize_height: type, optional
        lorem ipsum dolor
    ratio: type, optional
        lorem ipsum dolor
    levels: type, optional
        lorem ipsum dolor
    diagnose: type, optional
        lorem ipsum dolor
    mode: type, optional
        lorem ipsum dolor
    tag: type, optional
        lorem ipsum dolor
    
    Returns
    -------
    param : type
        Description
    """
    
    for f in fdata:
        print(f)
        success = True
        
        # read in scan data files
        parent = parentdir + '/' + f + '/'
        scans, hdata, hmaps, ims = readdata(parent,allowance=allowance,normalize_height=normalize_height)
        
        height,width,a = ims[0].shape
        im = ims[0]
        hm = hmaps[0]

        for i in range(len(scans)-1):
            hmn = hmaps[i+1]
            imn = ims[i+1]
            
            # find and match features
            if mode == 'sift':
                sift = cv.SIFT_create()
                kp, des = sift.detectAndCompute(im,None)
                kpn, desn = sift.detectAndCompute(imn,None)
            elif mode == 'orb':
                orb = cv.ORB_create(nfeatures=32000)
                kp, des = orb.detectAndCompute(im,None)
                kpn, desn = orb.detectAndCompute(imn,None)
                des = np.float32(des)
                desn = np.float32(desn)
            else:
                print('Mode is not correctly specified, try "sift" for best results')
                return

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50) # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des,desn,k=2)

            # ratio test as per Lowe's paper to filter for good matches
            dxy = {}
            sign = 0
            matchesMask = [[0,0] for i in range(len(matches))]
            for i,(m,n) in enumerate(matches):
                if m.distance < ratio*n.distance:
                    p = kp[m.queryIdx].pt
                    pn = kpn[m.trainIdx].pt
                    dy = int(pn[1] - p[1])
                    dx = int(pn[0] - p[0])
                    if dx != 0 and np.abs(dy) >= height/6 and p[1] < height:
                        dxy[p] = [(dx,dy),pn]
                        sign += dy/np.abs(dy)
                        matchesMask[i]=[1,0]
                        
            if diagnose:
                draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=cv.DrawMatchesFlags_DEFAULT)
                imx = cv.drawMatchesKnn(im,kp,imn,kpn,matches,None,**draw_params)
            
            bins, xedges, yedges = np.histogram2d([a[0] for a in dxy],[a[1] for a in dxy],bins=[[(width-1)/nbins * i for i in range(nbins+1)],[(height-1)/nbins * i for i in range((i+1)*nbins+1)]])
            bins = bins.T

            # find rectangle containing four highest density bins
            lims = denserect(nbins,bins)
            
            if diagnose:
                cv.rectangle(imx,(int(lims[2]*width/nbins),int(lims[0]*height/nbins)),(int((lims[3]+1)*width/nbins),int((lims[1]+1)*height/nbins)),(0,0,255),5) # draw rectangle on image
                plt.clf()
                plt.imshow(imx)
                plt.show()

            # compute statistics only within rectangle to help minimize noise
            region = []
            for p in dxy:
                if p[0] >= int(lims[2]*width/nbins) and p[0] <= int((lims[3]+1)*width/nbins) and p[1] >= int(lims[0]*height/nbins) and p[1] <= int((lims[1]+1)*height/nbins):
                    if dxy[p][0][1]/np.abs(dxy[p][0][1]) == sign/np.abs(sign) and len(p) == 2 and len(dxy[p][1]) == 2:
                        region.append((p,dxy[p][1],dxy[p][0]))

            src_pts = np.float32([a[0] for a in region]).reshape(-1, 1, 2)
            dst_pts = np.float32([a[1] for a in region]).reshape(-1, 1, 2)

            transformation_rigid_matrix, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts)
            border = cv.perspectiveTransform(np.array([[[0, im.shape[0]]], [[im.shape[1], im.shape[0]]]], dtype=np.float32), np.array([transformation_rigid_matrix[0], transformation_rigid_matrix[1], [0,0,1]], dtype=np.float32))
            ylim = np.max([p[0][1] for p in border])
            fixed = cv.warpAffine(im, transformation_rigid_matrix, (im.shape[1],int(ylim)))
            
            # blend overlapping region 
            fborder = cv.perspectiveTransform(np.array([[[0, 0]], [[im.shape[1], 0]]], dtype=np.float32), np.array([transformation_rigid_matrix[0], transformation_rigid_matrix[1], [0,0,1]], dtype=np.float32))
            flim = int(np.max([p[0][1] for p in fborder]))
            frange = height - flim - 1
            if frange <= 0 or flim < 0:
                print('Error matching scans')
                success = False
                break
            if levels:
                fixed[flim:height,:imn.shape[1]] = stitchborder(fixed[flim:height,:imn.shape[1]],imn[flim:height],levels=levels)
            else:
                for i in range(frange):
                    fixed[flim+i+1,:imn.shape[1]] = i/frange * fixed[flim+i+1,:imn.shape[1]] + (1-i/frange) * imn[flim+i+1]
            fixed[:flim,:imn.shape[1]] = imn[:flim]
            im = fixed

            fixedhm = cv.warpAffine(hm, transformation_rigid_matrix, (im.shape[1],int(ylim)))
            fixedhm[0:height,0:width] = hmn
            hm = fixedhm
            
            if diagnose:
                plt.clf()
                plt.imshow(im)
                plt.show()
                    
        if not diagnose:
            plt.clf()
            plt.imshow(im)
            plt.show()
            cv.imwrite('output/'+parent.split('/')[-2]+'_stitched'+tag+'.jpg',im)

            plt.clf()
            plt.imshow(hm,cmap='viridis')
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig('output/'+parent.split('/')[-2]+'_stitched'+tag+'_hm.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        
        else:
            plt.clf()
            plt.imshow(hm,cmap='viridis')
            plt.show()
        
        if success:
            print('Success')