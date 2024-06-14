import nrrd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import scipy

import SimpleITK as sitk 

# Normalize the whole image (get rid of HU index of air, and normalize voxel size)
def tranform_to_hu(image, intercept, rescale):
    return image * rescale + intercept

# Normalize the spacing of the whole image to isomorphic voxel size 1mm^3 
def resample_spacing(image, current_spacing, ideal_spacing = [[1,0,0],[0,1,0],[0,0,1]]):
    if np.array_equal(current_spacing,ideal_spacing):
        return image 

    resize_factor = np.array(current_spacing) / np.array(ideal_spacing)
    new_shape = np.round(image.shape * resize_factor) 
    new_resize_factor = np.array(new_shape) / np.array(image.shape)
    
    new_spacing = current_spacing / new_resize_factor

    new_resize_factor = [new_resize_factor[0][0], new_resize_factor[1][1], new_resize_factor[2][2]] # convert to a 1x3 list to avoid error for zoom function
    # Works but is slow
    image = scipy.ndimage.zoom(image, new_resize_factor, mode='nearest', order = 0) #nearest

    # Something with matlab, haven't tested    
    # image = imresize(image, new_resize_factor) #nearest 
    new_spacing = formatSpaceDirection(new_spacing)
    return image, new_spacing 


def window_image_to_adbomen(image, window_center, window_width):
    # image = sitk.GetArrayFromImage(image)
    img_max = window_center + int(window_width / 2)
    img_min = window_center - int(window_width / 2)
    return np.clip(image, img_min, img_max)

def formatSpaceDirection(spaceDirection):
    return np.array( [spaceDirection[0][0], spaceDirection[1][1], spaceDirection[2][2]]) 

def check_alignment(imageData, segmentData):
    """ A list of 3 elements, the space dimension, the space, and the space origin. Checks if the images are the same between each one"""
    valid = True
    if not np.array_equal( imageData[0], segmentData[0]):
        print(f"The space dimension is not the same, {imageData[0]} vs {segmentData[0]}")
        valid = False
    if imageData[1] != segmentData[1]:
        print(f'The space direction is not the same, {imageData[1]} vs {segmentData[1]}')
        valid = False
    if not np.array_equal(imageData[2],segmentData[2]):
        print(f'The space origin is not the same, {imageData[2]} vs {segmentData[2]}')
        valid = False
    return valid


def centerXYOfImage(overlay_mask, segment_mask, segmentedSlices, padding=25):
    """ 
    Centers the X and Y of the image to crop the image. segmentedSlices is given as an array of z-value slices because the same approach to x_indicies and y_indicies does not work on overlay_segment (works for x and y though)
    """
    x_indices, y_indices, _ = np.where(segment_mask == 1)
    # Get the bounding box for x and y dimensions
    min_x, max_x = x_indices.min(), x_indices.max()
    min_y, max_y = y_indices.min(), y_indices.max()

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    width = abs(max_x - min_x) // 2
    height = abs(max_y - min_y) // 2

    print('hi')    
    
    # Get the dimensions of the cropped image
    start_x = max(0, center_x - width - padding)
    end_x = min(segment_mask.shape[0], center_x + width + padding)
    start_y = max(0, center_y - height - padding)
    end_y = min(segment_mask.shape[1], center_y + height + padding)

    # # Adjust the crop region if it's smaller than 100x100
    # if end_x - start_x < 100:
    #     if start_x == 0:
    #         end_x = min(segment_mask.shape[0], 100)
    #     elif end_x == segment_mask.shape[0]:
    #         start_x = max(0, segment_mask.shape[0] - 100)
    # if end_y - start_y < 100:
    #     if start_y == 0:
    #         end_y = min(segment_mask.shape[1], 100)
    #     elif end_y == segment_mask.shape[1]:
    #         start_y = max(0, segment_mask.shape[1] - 100)

    print(start_x, end_x, start_y, end_y)

    segmentedSlices = np.sort(np.array(segmentedSlices))

    
    return overlay_mask[start_x:end_x, start_y:end_y, segmentedSlices[0]:segmentedSlices[len(segmentedSlices)-1]]

