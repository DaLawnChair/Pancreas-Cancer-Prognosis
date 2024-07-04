Current progress:
* Added code for a single slice model in imagSegmentationMultipleSingleSlice.ipynb
 

Need to do [current]:
* Apply image transformations on these images
* Choose fixed image dimension for all images to put into model 
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

