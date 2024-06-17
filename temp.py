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


def preprocess(wholeHeader, segmentHeader, verbose=True):
    error = False # Error flag to check if there was an error in the preprocessing

    # Check if the images are aligned
    wholeHeader, segmentHeader = makeAlign(wholeHeader, segmentHeader)
    imagesAreAligned = isAligned(wholeHeader, segmentHeader)
    print(f'Are the two images aligned now?: {imagesAreAligned}')

    if not imagesAreAligned:
        error = True
        return None, None, True
    
    # Set the spacing of the image to 1x1x1mm voxel spacing
    wholeHeader.SetSpacing([1,1,1])
    segmentHeader.SetSpacing([1,1,1])
    imagesSpacingAligned = wholeHeader.GetSpacing() == segmentHeader.GetSpacing() 
    print(f'Are the two images aligned in terms of spacing?: {imagesSpacingAligned}')

    if not imagesSpacingAligned:
        error = True
        return None, None, True
    

    imagesSizeAligned = wholeHeader.GetSize() == segmentHeader.GetSize() 
    print(f'Are the two images aligned in terms of size?: {imagesSizeAligned}')

    if not imagesSizeAligned:
        wholeHeader, segmentHeader = resampleSizes(wholeHeader, segmentHeader)
        print(f'whole size: {wholeHeader.GetSize()}')
        print(f'segment size: {segmentHeader.GetSize()}')
        imagesSizeAligned = wholeHeader.GetSize() == segmentHeader.GetSize() 
        print(f'Are the two images aligned in terms of size now?: {imagesSizeAligned}')
        if not imagesSizeAligned:
            error = True
            return None, None, True


    
    # Convert the images into numpy arrays for further processing, take the transpose as the format is z,y,x
    whole = sitk.GetArrayFromImage(wholeHeader).T
    segment = sitk.GetArrayFromImage(segmentHeader).T

    print(f'Spacing of whole:{whole.shape}')
    print(f'Spacing of segment:{segment.shape}')
    
    # Windowing parameters for the abdomen
    ABDOMEN_UPPER_BOUND = 215
    ABDOMEN_LOWER_BOUND = -135
    window_center = (ABDOMEN_UPPER_BOUND+ABDOMEN_LOWER_BOUND) / 2
    window_width = (ABDOMEN_UPPER_BOUND-ABDOMEN_LOWER_BOUND) / 2

    # Window and resample the whole image
    augmented_whole = window_image_to_adbomen(whole, window_center, window_width)

    # Resample the segment image to the same spacing as the whole image
    augmented_segment = segment

    # plt.imshow(augmented_whole[:, :, 53], cmap="gray")

    # Get the slice indices where the segment is present in 
    segmentedSlices = [] 
    for index in range(augmented_segment.shape[-1]):
        if len(np.unique(augmented_segment[:,:,index])) > 1:
            segmentedSlices.append(index)

    print(f'Segment slice indices:{segmentedSlices}')


    overlay_segment = augmented_whole * augmented_segment
    # print(overlay_segment.shape)
    # print("Dimension of the CT scan is:", image.shape)
    # plt.imshow(overlay_segment[:, :, 133], cmap="gray")

    """
    # croppedSegment = centerXYOfImage(overlay_segment,augmented_segment,segmentedSlices)
    croppedSegment = overlay_segment[:,:,segmentedSlices[0]:segmentedSlices[len(segmentedSlices)-1]+1]
    # croppedSegment = window_image_to_adbomen(croppedSegment, window_center, window_width)
    # croppedSegment[croppedSegment<0]=0 # Window the image so that the background is completely black for all slices

    # croppedSegment = convertNdArrayToCV2Image(croppedSegment)

    if verbose:
        print(f'CroppedSegment shape: {croppedSegment.shape}')
        # Display the segmented image slices 

        columnLen = 10
        rowLen = max(2,croppedSegment.shape[-1] // columnLen + 1) 
        figure,axis = plt.subplots( rowLen, columnLen, figsize=(10, 10))
        
        rowIdx = 0
        for idx in range(croppedSegment.shape[-1]):        
            if idx%columnLen == 0 and idx>0:
                rowIdx += 1
            axis[rowIdx][idx%columnLen].imshow(croppedSegment[:,:,idx], cmap="gray")
            axis[rowIdx][idx%columnLen].axis('off')

        # Turn off the axis of the rest of the subplots
        for i in range(idx+1, rowLen*columnLen):
            if i%columnLen == 0:
                rowIdx += 1
            axis[rowIdx][i%columnLen].axis('off')
        
        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    return whole, croppedSegment, error
"""


"""
# baseFilepath = 'PDAC-Response/PDAC-Response/ImagingData/Pre-treatment/CASE481_empty'
# segmentFilePath = baseFilepath + '/CASE481_BASE_PRT_TUM_CV.seg.nrrd'
# wholeFilePath = baseFilepath + '/CASE481_BASE_PRT_WHOLE_CT.nrrd' 
# whole, wholeHeader = nrrd.read(wholeFilePath)
# segment, segmentHeader = nrrd.read(segmentFilePath)

allFolders = ['CASE244', 'CASE246', 'CASE247', 'CASE251', 'CASE254', 'CASE256', 'CASE264', 'CASE265', 'CASE270', 'CASE272', 'CASE274', 
              'CASE467', 'CASE468', 'CASE470', 'CASE471', 'CASE472', 'CASE479', 'CASE480', 'CASE482', 'CASE484', 'CASE485', 'CASE494', 'CASE496', 'CASE499', 
              'CASE500', 'CASE505', 'CASE515', 'CASE520', 'CASE523', 'CASE531', 'CASE533', 'CASE534', 'CASE535', 'CASE541', 'CASE543', 'CASE546', 'CASE547', 'CASE548', 'CASE549', 'CASE550', 'CASE551', 'CASE554', 'CASE555', 'CASE557', 'CASE559', 'CASE560', 'CASE562', 'CASE563', 'CASE565', 'CASE568', 'CASE569', 'CASE572', 'CASE574', 'CASE575', 'CASE577', 'CASE578', 'CASE580', 'CASE581', 'CASE585', 'CASE586', 'CASE587', 'CASE588', 'CASE589', 'CASE593', 'CASE594', 'CASE596', 'CASE598', 
              'CASE600', 'CASE602', 'CASE603', 'CASE604', 'CASE605', 'CASE608', 'CASE610', 'CASE611', 'CASE615', 'CASE616', 'CASE622', 'CASE623', 'CASE624', 'CASE630', 'CASE632', 'CASE635']
alreadySeem=['CASE244', 'CASE246', 'CASE247', 'CASE251', 'CASE254', 'CASE256', 'CASE264', 'CASE265', 'CASE270', 'CASE272', 'CASE274', 
              'CASE467', 'CASE468', 'CASE470', 'CASE471', 'CASE472', 'CASE479', 'CASE480', 'CASE482', 'CASE484', 'CASE485', 'CASE494', 'CASE496', 'CASE499', 
              'CASE500', 'CASE505', 'CASE515', 'CASE520']
# alreadySeem = ['CASE244', 'CASE246', 'CASE247', 'CASE251', 'CASE254', 'CASE256', 'CASE263', 'CASE264', 'CASE265', 'CASE270', 'CASE272', 'CASE274', 
#  'CASE467', 'CASE468', 'CASE470', 'CASE471', 'CASE472', 'CASE476', 'CASE479', 'CASE480', 'CASE481_empty', 'CASE482', 'CASE484', 'CASE485', 'CASE488', 'CASE494', 'CASE496', 'CASE499', 
#  'CASE500', 'CASE505', 'CASE515', 'CASE520', 'CASE523', 'CASE525', 'CASE531', 'CASE533', 'CASE534', 'CASE535', 'CASE537', 'CASE539', 'CASE541', 'CASE543', 'CASE546', 'CASE547', 'CASE548', 'CASE549', 'CASE550', 'CASE551', 'CASE554', 'CASE555', 'CASE557', 'CASE559', 'CASE560', 'CASE562', 'CASE563', 'CASE564', 'CASE565', 'CASE568', 'CASE569', 'CASE572', 'CASE574', 'CASE575', 'CASE577', 'CASE578', 'CASE580', 'CASE581', 'CASE585', 'CASE586', 'CASE587', 'CASE588', 'CASE589', 'CASE593', 'CASE594', 'CASE596', 'CASE598', 
#  'CASE600', 'CASE601', 'CASE602', 'CASE603', 'CASE604', 'CASE605', 'CASE608', 'CASE610', 'CASE611', 'CASE615', 'CASE616', 'CASE621', 'CASE622', 'CASE623', 'CASE624', 'CASE629']
baseFilepath = 'PDAC-Response/PDAC-Response/ImagingData/workingdir/'

successes=0
count=0
for folder in os.listdir(baseFilepath):

    # if folder in alreadySeem:
    #     continue 
    count = 0
    # for file in os.listdir(os.path.join(baseFilepath,folder)):
        
    #     if 'segmentation' in file or 'segmention' in file: 
    #         count+=1
    #         postSegmentHeader = sitk.ReadImage(os.path.join(baseFilepath,folder,file))
    #         # postSegment = sitk.ReadImage(os.path.join(baseFilepath,folder,file))

    #     elif 'TUM' in file or 'SMV' in file:
    #         count+=1
    #         # segment, segmentHeader = nrrd.read(os.path.join(baseFilepath,folder,file))
    #     elif file.endswith('CT.nrrd'):
    #         count+=1
    #         wholeHeader = sitk.ReadImage(os.path.join(baseFilepath,folder,file))
    
    print('==============================================================')
    print(folder, count==3)
      
    # # For a single case
    postSegmentHeader = sitk.ReadImage(os.path.join(baseFilepath,'CASE523/546173_Tu_segmentation_NH.seg.nrrd'))
    wholeHeader = sitk.ReadImage(os.path.join(baseFilepath,'CASE523/CASE523_BASE_PRT_WHOLE_CT.nrrd'))

    whole, croppedSegment,error = preprocess(wholeHeader, postSegmentHeader, verbose=True) 
    if error:
        print('Error in preprocessing')
        # successes += 1
        continue
    # count+=1

# print('Successes:', successes)
# print('Total processed:', count)
    
    
"""