Current progress:
* Fixed CASE265, CASE535, and CASE578 with having empty segmentations within the segmentation range
* Allowed ability to change verboseness of preprocess()
* Added view of the whole CT overlayed with the mask in preprocess()
* Removed changes most functionality with post-treatment images as they cause lots of issues
* Added imageSegmentationSingleEval.ipynb, which enables single case evaluation.

Need to do:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546
* Apply image transformations on these images
* Categorize data based on the .xsxl file, analyzing potential train-test splits
* Choose fixed image dimension for all images to put into model 