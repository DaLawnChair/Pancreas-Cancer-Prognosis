import nrrd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import scipy

import SimpleITK as sitk 

def window_image_to_adbomen(image, window_center, window_width):
    # image = sitk.GetArrayFromImage(image)
    img_max = window_center + int(window_width / 2)
    img_min = window_center - int(window_width / 2)
    return np.clip(image, img_min, img_max)

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


    segmentedSlices = np.sort(np.array(segmentedSlices))

    
    return overlay_mask[start_x:end_x, start_y:end_y, segmentedSlices[0]:segmentedSlices[len(segmentedSlices)-1]+1]

