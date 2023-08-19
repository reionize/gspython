import os
import sys
import numpy as np
import scipy
import random
import cv2 as cv
from matplotlib import pyplot as plt

sys.path.append(r"../")
import readscan
import readtmd

def readdata(parent,allowance=10,baseline=75,normalize_height=True,reverse_order=False):
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
        Option to normalize heightmap by sampling same corner regions and subtracting out the resulting bilinear gradient. Default is True.
    reverse_order: bool, optional
        Determines which order the scans are placed in. Default is False, assumes that the first scan is the bottom frame and each scan is stacked on top.
    
    Returns
    -------
    scans : list <str>
        List of scan directories
    hmaps : list <numpy.ndarray>
        List of ndarrays containing height data at every pixel location in each scan, represented from 0-1. 
    ims : list <numpy.ndarray>
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
                scans.append(name)
    scans.sort(reverse=reverse_order)

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
        ims.append(subtractgrad(im,baseline=baseline,allowance=allowance))

        # normalize gradient for heightmap if option is turned on        
        hm = hdata[i][0]
        if normalize_height:
            hm = subtractgrad(hm,baseline=0,allowance=allowance)
        hmaps.append(hm)
        
    return scans,hmaps,ims

def subtractgrad(im, baseline=0, allowance=10):
    """
    Normalizes image for background variation by subtracting out bilinear gradient from image corners.
    
    Parameters
    ----------
    im : numpy.ndarray
        Input image to be normalized, in a standard OpenCV format with either a float between 0-1 or a list of three integers in RGB representing each pixel.
    allowance : int, optional
        Width parameter in pixels which determines size of corner regions to sample from when subtracting out bilinear gradient to minimize variation in background brightness. Default is 10.
    baseline: int, optional
        Baseline brightness parameter between 0-255 which is added to entire image. Default is 0.
    
    Returns
    -------
    result : numpy.ndarray
        Normalized image in same format as input.
    """
    
    height,width = im.shape[:2]
    a = 1
    if len(im.shape) > 2:
        a = im.shape[-1]
    
    # compute average value of corners
    corners = [np.average(im[:allowance, :allowance], axis = (0,1)),np.average(im[:allowance, -allowance:], axis = (0,1)),np.average(im[-allowance:, -allowance:], axis = (0,1)),np.average(im[-allowance:, :allowance], axis = (0,1))]
    
    # adjust for baseline value
    if a == 1:
        gradient = np.asarray([[corners[0]*255,corners[1]*255],[corners[3]*255,corners[2]*255]])-[[baseline,baseline],[baseline,baseline]]
    else:
        gradient = np.floor([[corners[0],corners[1]],[corners[3],corners[2]]])-[[[baseline]*a,[baseline]*a],[[baseline]*a,[baseline]*a]]
    
    # compute mask
    mask = cv.resize(gradient,None,fx=width/2,fy=height/2,interpolation=cv.INTER_LINEAR)
    if a == 1:
        result = im - mask/255
    else:
        im = im-mask
        im = im.clip(0,255).astype(int)
        result = cv.normalize(im, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        
    return result

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

def stitchborderpyr(im1, im2, levels=5, reverse=False):
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
    reverse: bool, optional
        Determines which image to stack on top. Default is False, which stacks im2 on top of im1.
    
    Returns
    -------
    result : numpy.ndarray
        Vertically blending image with im2 on top of im1 (or vice versa if reverse is True), in the same format as the inputs.
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
        ls = []
        if reverse:
            ls = np.vstack((l1[0:rows//2], l2[rows//2:]))
        else:
            ls = np.vstack((l2[0:rows//2], l1[rows//2:]))
        LS.append(ls)
    result = LS[0]
    for i in range(1,levels):
        b = LS[i].shape
        result = cv.pyrUp(result,dstsize=(b[1],b[0]))
        result = cv.add(result, LS[i])
        
    return result

def getmatches(im1, im2, ratio=0.7, mode='sift'):
    """
    Detects and computes matches between features in the input images. 
    
    Parameters
    ----------
    im1: numpy.ndarray
        First image to be matched, in a standard OpenCV format.
    im2: numpy.ndarray
        Second image to be matched, in the same format.
    ratio: float, optional
        Cutoff to use for filtering good matches, as per Lowe's paper. Default is 0.7.
    mode: string, optional
        Determines what algorithm to use for feature detectiong and matching. Default is 'sift' (recommended), but also accepts 'orb'.
    
    Returns
    -------
    kp1 : tuple <cv.KeyPoint>
        Tuple of keypoint objects in the first image.
    kp2 : tuple <cv.KeyPoint>
        Tuple of keypoint objects in the second image.
    matches : tuple
        Tuple of 2-tuples which stores the two best matches for each keypoint in the first image.
    matchesMask : list
        Mask which selects "good" matches, as determined by Lowe's ratio test.
    dxy : dict
        Dictionary mapping coordinates of keypoints in im1 (x1, y1) to the coordinates of the matched keypoints in im2 and the difference between the coordinates in the form [(dx, dy), (x2, y2)].
    sign : float
        Diagnostic parameter which is negative if the majority of matches have negative slope and positive if the majority of matches have positive slope.
    """
    height,width,_ = im2.shape
    
    # find and match features
    if mode == 'sift':
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(im1,None)
        kp2, des2 = sift.detectAndCompute(im2,None)
    elif mode == 'orb':
        orb = cv.ORB_create(nfeatures=32000)
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        des1 = np.float32(des)
        des2 = np.float32(desn)
    else:
        print('Mode is not correctly specified, try "sift" for best results')
        return

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # ratio test as per Lowe's paper to filter for good matches
    dxy = {}
    sign = 0
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            p = kp1[m.queryIdx].pt
            pn = kp2[m.trainIdx].pt
            dy = int(pn[1] - p[1])
            dx = int(pn[0] - p[0])
            if dx != 0 and np.abs(dy) >= height/6 and p[1] < height:
                dxy[p] = [(dx,dy),pn]
                sign += dy/np.abs(dy)
                matchesMask[i]=[1,0]
                
    return kp1, kp2, matches, matchesMask, dxy, sign

def stitchscans(objects, allowance=20, normalize_height=True, ratio=0.7, nbins=10, levels=0, reverse_order=False, diagnose=False, mode='sift', tag=''):
    """
    Reads and stitches together scans of objects in fdata. Note that when using diagnostic mode, stitching process for each image may be paused until current diagnostic image is closed, as cv.waitKey() is not specified.
    
    Parameters
    ----------
    objects: list <int>
        List of strings specifying folders for each object to stitch together.
    allowance: int, optional
        Width parameter in pixels which determines size of corner regions to sample from when subtracting out background brightness in photograph. Default is 20.
    normalize_height: bool, optional
        Option to normalize heightmap by sampling same corner regions and subtracting out the resulting  gradient. Default is True.
    ratio: float, optional
        Cutoff to use for filtering good matches, as per Lowe's paper. Default is 0.7.
    levels: int, optional
        Number of levels to use in Gaussian pyramids blending method. Default is 0, which applies linear blending to overlapping regions.
    reverse_order: bool, optional
        Determines which order the scans are placed in. Default is False, assumes that the first scan is the bottom frame and each scan is stacked on top.
    diagnose: bool, optional
        Option to use diagnostic mode, which shows additional intermediate steps and does not save final output.
    mode: str, optional
        Algorithm to use for detecting and computing matches between keypoints in scan images. Default is 'sift' (recommended), but also accepts 'orb'.
    tag: str, optional
        Tag to include in output filename.
    
    Returns
    -------
    None
    """
    
    for parent in objects:
        if parent[-1] != '/':
            parent += '/'
        f = parent.split('/')[-2]
        print('Processing: \t',f)
        success = True
        
        # read in scan data files
        scans, hmaps, ims = readdata(parent,allowance=allowance,normalize_height=normalize_height,reverse_order=reverse_order)
        
        height,width,_ = ims[0].shape
        im = ims[0]
        hm = hmaps[0]

        # iterate through pairs of consecutive images to stitch together
        for i in range(len(scans)-1):
            hmn = hmaps[i+1]
            imn = ims[i+1]
            
            # compute best feature matches
            kp, kpn, matches, matchesMask, dxy, sign = getmatches(im,imn,ratio=ratio,mode=mode)
                        
            if diagnose:
                draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=cv.DrawMatchesFlags_DEFAULT)
                imx = cv.drawMatchesKnn(im,kp,imn,kpn,matches,None,**draw_params)
            
            bins, xedges, yedges = np.histogram2d([a[0] for a in dxy],[a[1] for a in dxy],bins=[[(width-1)/nbins * i for i in range(nbins+1)],[(height-1)/nbins * i for i in range((i+1)*nbins+1)]])
            bins = bins.T

            # find rectangle containing four highest density bins
            lims = denserect(nbins,bins)
            
            # diagnose matches
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

            # determine least squares affine transformation to stitch together scans
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
            
            # show diagnostics
            if diagnose:
                plt.clf()
                plt.imshow(im)
                plt.show()
        
        # write output images
        if not diagnose:
            if len(tag) > 0:
                tag = '_' + tag
            
            plt.clf()
            plt.imshow(im)
#             plt.show()
            cv.imwrite('output/'+parent.split('/')[-2]+'_stitched'+tag+'.jpg',im)

            plt.clf()
            plt.imshow(hm,cmap='viridis')
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig('output/'+parent.split('/')[-2]+'_stitched'+tag+'_hm.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        
        # show diagnostics
        else:
            plt.clf()
            plt.imshow(hm,cmap='viridis')
            plt.show()
        
        # confirm that no warnings were triggered during algorithm run
        if success:
            print('Success')