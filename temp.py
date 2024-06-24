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

# postSegmentHeader = sitk.ReadImage('PDAC-Response/PDAC-Response/ImagingData/workingdir/CASE523/546173_Tu_segmentation_NH.seg.nrrd')
# postSegment = sitk.GetArrayFromImage(postSegmentHeader)
# print(postSegmentHeader.GetSize())
# print(type(postSegmentHeader))
# print(type(postSegment))

# print(np.unique(postSegment))

# postSegment = postSegment.T
# print(postSegment.shape)

# segmentedSlices = [] 
# for index in range(postSegment.shape[-1]):
#     if len(np.unique(postSegment[:,:,index])) > 1:
#         segmentedSlices.append(index)

# print(segmentedSlices)


# plt.imshow(postSegment[:,:,53], cmap="gray")



original = sitk.read("pancreas_data/pancreas_data/neoadjuvant_pdac/01/01_neo_pdac_pre_Tumor.seg.nrrd")
originalArray = sitk.GetArrayFromImage(original).T
originalArray[originalArray == -1000] = 0
originalArray[originalArray > 1] = 1
originalArray[originalArray <0] = 1
modified = sitk.GetImageFromArray(originalArray)
modified.CopyInformation(original)

sitk.WriteImage(modified, "lol.seg.nrrd")

print()

print(np.unique(original))