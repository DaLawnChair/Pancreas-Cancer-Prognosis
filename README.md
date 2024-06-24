Current progress:
* Copied over the functions from imageSubtraction.py to imageSegmentaitonMultiple, deleleted the .py 
* Refractor the preproccess function to multiple individual functions
* Make PDAC-Response_working.xsxl only have 2 columns, the case name and the recist criteria
* Removed the 'throw-away' of the values of croppedSegment[croppedSegment<0.0001] = 0 as it is not needed after further inspection
* Removed cases from the xsxl that have fewer than 7 slices as well as the corrupted pre-treatment cases CASE533 and CASE629

Need to do
* Apply image transformations on these images
* Choose fixed image dimension for all images to put into model 
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

