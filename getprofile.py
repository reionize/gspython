import os
import numpy as np
import struct
import argparse
import cv2

def getprofile(heightMap, res, x1, y1, x2, y2, normalized=False):
    p1 = np.array((x1, y1))
    p2 = np.array((x2, y2))
    dp = p2 - p1
    len = np.linalg.norm(dp)
    u = np.array((-dp[1], dp[0]))
    u /= len
    samples = []
    nPoints = int(np.ceil(len)+1)
    print(nPoints)
    np_ = nPoints-1
    for i in range(nPoints):
        tv = i/np_
        pt = np.array((x1+tv*dp[0], y1+tv*dp[1]))
        samples.append([pt, u, tv*len])

    profiles = []
    curr_tv = 0
    lastp = samples[0][0]
    
    min_z = 1e10
    max_z = -1e10

    for sample in samples:
        pt = sample[0]
        dx = pt[0] - lastp[0]
        dy = pt[1] - lastp[1]
        dt = np.sqrt(max(dx*dx + dy*dy, 0.0))
        curr_tv += dt
        zv = interpbicubic(heightMap, pt[1], pt[0])
        p = np.array((int(curr_tv)*res, zv))
        profiles.append(p)
        lastp = pt

        if min_z>p[1]:
            min_z = p[1]
        if max_z<p[1]:
            max_z = p[1]
    
    if normalized:
        normalized_profile = [np.array((p[0], (p[1]-min_z)/(max_z-min_z))) for p in profiles]
        profiles = normalized_profile

    return profiles

def cubickernel(t):
    k = 0.0
    tv = abs(t)
    
    if tv <= 1.0:
        k = ((1.5*tv - 2.5)*tv)*tv + 1.0
    elif tv <= 2.0:
        k = ((-0.5*tv + 2.5)*tv - 4.0)*tv + 2.0
    
    return k

def interpbicubic(image, yy, xx):
    xdim = image.shape[1]
    ydim = image.shape[0]

    #  Clamp to image bounds
    if (xx < 0.0):
        xx = 0.0

    if (yy < 0.0):
        yy = 0.0

    if (xx >= xdim-1):
        xx = xdim-1

    if (yy >= ydim-1):
        yy = ydim-1

    #  f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
    xb = int(np.floor(xx))
    yb = int(np.floor(yy))

    xt = xx - xb
    yt = yy - yb

    # Extract 4 x 4 region
    crop = np.zeros((4,4), dtype='float')
    for ii in range(-1, 3):
        for jj in range(-1, 3):
            yix = yb + ii
            xix = xb + jj

            val = 0.0
            #  Both in bounds
            if (xix >= 0 and xix < xdim and yix >= 0 and yix < ydim):
                val = image[yix, xix]

            #  X is in bounds, but not y
            elif (xix >= 0 and xix < xdim):
                if (yix < 0):
                    val = image[2, xix] - 3*image[1, xix] + 3*image[0, xix]
                else:
                    val = 3*image[ydim-1,xix] - 3*image[ydim-2,xix] + image[ydim-3,xix]

            #  Y is in bounds, but not x
            elif ( yix >= 0 and yix < ydim):
                if (xix < 0):
                    val = image[yix, 2] - 3*image[yix, 1] + 3*image[yix, 0]
                else:
                    val = 3*image[yix, xdim-1] - 3*image[yix, xdim-2] + image[yix,xdim-3]
            crop[ii+1, jj+1] = val

    #  Set corners (if necessary)
    if (yb == 0 and xb == 0):
        val = 3*crop[1, 0] - 3*crop[2, 0] + crop[3, 0]
        crop[0, 0] = val

    if (yb == ydim-2 and xb == 0):
        val = 3*crop[2, 0] - 3*crop[1, 0] + crop[0, 0]
        crop[3, 0] = val

    if (yb == 0 and xb == xdim-2):
        val = 3*crop[0, 2] - 3*crop[0, 1] + crop[0, 0]
        crop[0, 3] = val

    if (yb == ydim-2 and xb == xdim-2):
        val = 3*crop[2, 3] - 3*crop[1, 3] + crop[0, 3]
        crop[3, 3] = val

    kv = 0.0
    for ii in range(0, 4):
        ky = cubickernel(ii-1.0-yt)
        for jj in range(0, 4):
            kx = cubickernel(jj-1.0-xt)
            kv += kx*ky*crop[ii,jj]

    return kv