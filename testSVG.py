# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:36:17 2021

@author: oacom
"""


from svgpathtools import svg2paths2, Line, Path
import matplotlib.pyplot as plt
import numpy as np
import imageio

rshape = imageio.imread('road_shape.png')
plt.imshow(rshape)

paths, attributes, svg_attributes = svg2paths2('road_shape.svg')
P = paths[0]

for b in P:
    bp = b.point(np.linspace(0,1))
    bpx = np.real(bp)
    bpy = np.imag(bp)
    plt.plot(bpx,bpy)
    
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

op1 = offset_curve(P,100,20)

for b in op1:
    bp = b.point(np.linspace(0,1,10))
    bpx = np.real(bp)
    bpy = np.imag(bp)
    plt.plot(bpx,bpy)
    
op2 = offset_curve(P.reversed(),100,20)

for b in op2:
    bp = b.point(np.linspace(0,1,10))
    bpx = np.real(bp)
    bpy = np.imag(bp)
    plt.plot(bpx,bpy)