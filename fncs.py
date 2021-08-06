# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 22:08:43 2021

@author: oacom
"""


import numpy as np
from scipy.special import comb

def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

def smoothclamp(x, mi, mx): return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

def sigmoid(x,mi, mx): return mi + (mx-mi)*(lambda t: (1+200**(-t+0.5))**(-1) )( (x-mi)/(mx-mi) )


if __name__ == "__main__":
    x1 = np.linspace(-5,5)
    x2 = np.linspace(5,-5)
    
    y1 = smoothclamp(x1,min(x1),max(x1))
    y2 = smoothclamp(x2,min(x2),max(x2))