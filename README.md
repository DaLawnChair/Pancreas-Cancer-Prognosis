Current progress:
* Added code to drop all cases that have slice thickness less than 13
* getLargestSlice() now returns an dictionary of the index and an dictionary with sliceNumber and index in ascending order
* updateSlices() now can delete values of slices where the # of slices is greater than the param desiredNumberOfSlices
* Validated that the slices for >13 work with updateSlices() 
* Added documentation to updateSlices() where we duplicate slices

Need to do [current]:
* Apply image transformations on these images
* Choose fixed image dimension for all images to put into model 
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

