import nrrd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import scipy

import SimpleITK as sitk 
import os 

# import ipynb 

# import ipynb.fs.full.imageSubtraction 
from imageSubtractionFunctions import *
# %run imageSubtraction.ipynb


postSegmentHeader = sitk.ReadImage('PDAC-Response/PDAC-Response/ImagingData/workingdir/CASE523/546173_Tu_segmentation_NH.seg.nrrd')
postSegment = sitk.GetArrayFromImage(postSegmentHeader)
print(postSegmentHeader.GetSize())
print(type(postSegmentHeader))
print(type(postSegment))

print(np.unique(postSegment))

postSegment = postSegment.T
print(postSegment.shape)

segmentedSlices = [] 
for index in range(postSegment.shape[-1]):
    if len(np.unique(postSegment[:,:,index])) > 1:
        segmentedSlices.append(index)

print(segmentedSlices)


# plt.imshow(postSegment[:,:,53], cmap="gray")