import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import scipy

import SimpleITK as sitk 
import os 

# import ipynb 

# import ipynb.fs.full.imageSubtraction 

postSegmentHeader = sitk.ReadImage('Pre-treatment-only-pres/CASE585/CASE585_BASE_PRT_TUM_HK.seg.nrrd')
postSegment = sitk.GetArrayFromImage(postSegmentHeader)
print(postSegmentHeader.GetSize())
print(postSegmentHeader.GetDirection())
space_direction = postSegmentHeader.GetDirection()

if space_direction[8] < 0:
    image_array = np.flip(postSegment, axis=0)

def display_axial_slice(image_array, slice_index):
    axial_slice = image_array[slice_index, :, :]
    plt.imshow(axial_slice, cmap='gray')
    plt.title(f'Axial Slice {slice_index}')
    plt.axis('off')
    plt.show()


segmentedSlices = [] 
for index in range(postSegment.shape[-1]):
    if len(np.unique(postSegment[:,:,index])) > 1:
        segmentedSlices.append(index)
print(segmentedSlices)

axial_slice_index = image_array.shape[0] // 2
display_axial_slice(image_array, segmentedSlices[0])
display_axial_slice(image_array, segmentedSlices[2])
display_axial_slice(image_array, segmentedSlices[6])

# postSegment = postSegment.T
# print(postSegment.shape)



# plt.imshow(postSegment[:,:,53], cmap="gray")



# original = sitk.read("pancreas_data/pancreas_data/neoadjuvant_pdac/01/01_neo_pdac_pre_Tumor.seg.nrrd")
# originalArray = sitk.GetArrayFromImage(original).T
# originalArray[originalArray == -1000] = 0
# originalArray[originalArray > 1] = 1
# originalArray[originalArray <0] = 1
# modified = sitk.GetImageFromArray(originalArray)
# modified.CopyInformation(original)

# sitk.WriteImage(modified, "lol.seg.nrrd")

# print()

# print(np.unique(original))