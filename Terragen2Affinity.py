# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 08:51:53 2021

@author: oacom
"""

import imageio
from PIL import Image
import numpy as np

# Read Terrain
mat = imageio.imread('./Pass/TerrainInPass.exr')

# Normalize Terrain
mat_norm = (mat + np.abs(np.min(mat)))/ np.max(mat + np.abs(np.min(mat)))
mat_gray = mat_norm*255

img = Image.fromarray(np.uint8(mat_gray), 'L')
img.show()
img.save('./Pass/TerrainPic.jpg',"JPEG")