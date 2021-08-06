# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:49:12 2021

@author: oacom
"""

import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import ndimage
from fncs import sigmoid, smoothclamp
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from skimage.draw import polygon
from svgpathtools import svg2paths2, Line, Path
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import gaussian_filter
from progressbar import progressbar

# Settings
road_width = 25
border_width = 100

# Read Terrain
mat = imageio.imread('terrainIn2.exr')
mat0 = imageio.imread('terrainIn2.exr')
print(mat.shape)
# Creates PIL image
# img = Image.fromarray(np.uint8(mat), 'L')
# img.show()

# Read Road path pixels
rshape = imageio.imread('road_shape.png')
print(mat.shape)
plt.imshow(rshape)

# Read Road path vector
paths, attributes, svg_attributes = svg2paths2('road_shape.svg')
P = paths[0]

# Get pixels on vector
rx = []
ry = []
rnx = []
rny = []
for b in P:
    bp = b.point(np.linspace(0,1,400))
    bpx = np.real(bp)
    bpy = np.imag(bp)
    # plt.plot(bpx,bpy)
    rx.append(bpx)
    ry.append(bpy)
    # for t in np.linspace(0,1,200):
    #     bn = b.normal(t)
    #     bnx = np.real(bn)
    #     bny = np.imag(bn)
    #     # plt.plot(bpx,bpy)
    #     rnx.append(bnx)
    #     rny.append(bny)

rx = np.concatenate(rx)
ry = np.concatenate(ry)
# rnx = np.array(rnx)
# rny = np.array(rny)
road = (np.c_[rx,ry]).astype(np.int)
# road_n = np.c_[rnx,rny]

rmask = np.zeros(rshape.shape)
rmask[road[:,1],road[:,0]] = 1
plt.imshow(rmask)

# Get more detailed centerline for export
rc_x = []
rc_y = []
for b in P:
    bp = b.point(np.linspace(0,1,1000))
    bpx = np.real(bp)
    bpy = np.imag(bp)
    rc_x.append(bpx)
    rc_y.append(bpy)

rc_x = np.concatenate(rc_x).astype(np.int)
rc_y = np.concatenate(rc_y).astype(np.int)
rcmask = np.zeros(rshape.shape)
rcmask[rc_y,rc_x] = 1

# Extract path altitudes
pathval = mat[road[:,1],road[:,0]]

roadsmth = ndimage.gaussian_filter1d(pathval, 150)

def offset_curve(path, offset_distance, steps=1000):
    """Takes in a Path object, `path`, and a distance,
    `offset_distance`, and outputs an piecewise-linear approximation 
    of the 'parallel' offset curve."""
    nls = []
    for seg in path:
        # ct = 1
        for k in range(steps):
            t = k / steps
            offset_vector = offset_distance * seg.normal(t)
            nl = Line(seg.point(t), seg.point(t) + offset_vector)
            nls.append(nl)
    connect_the_dots = [Line(nls[k].end, nls[k+1].end) for k in range(len(nls)-1)]
    if path.isclosed():
        connect_the_dots.append(Line(nls[-1].end, nls[0].end))
    offset_path = Path(*connect_the_dots)
    return offset_path

# Get road surface
rsur = []
border = []
rbx = []
rby = []
bmask = np.zeros(rmask.shape)
fmask = np.zeros(rmask.shape)
rbmask = np.zeros(rmask.shape)

rL = offset_curve(P,road_width/2,15)
rR = offset_curve(P,-road_width/2,15)
rLb = offset_curve(P,road_width*1.3/2,15)
rRb = offset_curve(P,-road_width*1.3/2,15)
rLL = offset_curve(P,border_width/2,15)
rRR = offset_curve(P,-border_width/2,15)

for i in progressbar(range(len(rL))):
    # Shoulder
    bp1 = rLL[i].point(np.linspace(0,1,3))
    bpx1 = np.real(bp1)
    bpy1 = np.imag(bp1)
    bp2 = rRR[i].point(np.linspace(0,1,3))
    bpx2 = np.flip(np.real(bp2))
    bpy2 = np.flip(np.imag(bp2))
    rsy,rsx = polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2]))
    rss = np.c_[rsx,rsy]
    bmask[rsx,rsy] = 1
    rbmask[rsx,rsy] = 1
    
    # Road Surface
    bp1 = rL[i].point(np.linspace(0,1,3))
    bpx1 = np.real(bp1)
    bpy1 = np.imag(bp1)
    bp2 = rR[i].point(np.linspace(0,1,3))
    bpx2 = np.flip(np.real(bp2))
    bpy2 = np.flip(np.imag(bp2))
    rsy,rsx = polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2]))
    rs = np.c_[rsx,rsy]
    rmask[rsx,rsy] = 1
    rbmask[rsx,rsy] = 1
    
    # Get shoulder distances to center line
    dist_s = distance.cdist(np.c_[rss[:,1],rss[:,0]],road,'euclidean')
    idx_min = np.argmin(dist_s,axis=1)
    dist_min = np.min(dist_s,axis=1)
    clamped = smoothclamp(dist_min,road_width/2,max(dist_min))
    clamped_scaled = minmax_scale(clamped)
    clipped = dist_min
    clipped[clipped < road_width/2] = road_width/2
    clip_scale = minmax_scale(clipped)
    fade = 1-clamped_scaled
    fmask[rss[:,0],rss[:,1]] = fade
    
    # Get road surface distances to center line
    dist_r = distance.cdist(np.c_[rs[:,1],rs[:,0]],road,'euclidean')
    
    # Heights for road surface
    cloInd = np.argsort(dist_r,axis=1)
    hvals = np.mean(roadsmth[cloInd[:,0:3]],axis=1)
    
    # Apply shoulder
    mat[rss[:,0],rss[:,1]] = roadsmth[idx_min]*fade + mat[rss[:,0],rss[:,1]]*(1-fade)
    
    # Apply road surface
    mat[rsx,rsy] = hvals
    
    
plt.imshow(rmask)

# Smooth road surface and shoulder
mat_blur = gaussian_filter(mat,sigma=4)
mat = np.where(rbmask>0,mat_blur,mat)
mix_mask = np.maximum(fmask,rmask)
mat_out = mat*fmask + mat0*(1-fmask)

# Write Terrain
imageio.imwrite('terrainOut_curveRoad4.tif',mat_out.astype(np.float32))

# Write Masks
imageio.imwrite('terrainOut_curveRoad2_rcmask.png',(rcmask*255).astype(np.uint8))
imageio.imwrite('terrainOut_curveRoad2_rmask.png',(rmask*255).astype(np.uint8))
imageio.imwrite('terrainOut_curveRoad2_bmask.png',(bmask*255).astype(np.uint8))
imageio.imwrite('terrainOut_curveRoad2_fmask.png',(fmask*255).astype(np.uint8))
print('Done')

# Normalize Terrain
plt.figure()
mat_norm = (mat + np.abs(np.min(mat)))/ np.max(mat + np.abs(np.min(mat)))
mat_gray = mat_norm*255
plt.imshow(mat_gray)
