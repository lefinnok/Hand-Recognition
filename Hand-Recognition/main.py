# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:40:43 2021

@author: lefin
"""

import cv2
import numpy as np

misk = cv2.imread('miskatonic.jpg')

print(misk.shape)


cv2.imshow('img',cv2.resize(misk,dsize = (500,375), interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()
