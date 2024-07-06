Current progress:
* Added code for a single slice model in imagSegmentationMultipleSingleSlice.ipynb
* Updated code to work on a python3.8.10 virtual environment as opposed to conda so it can be run on purely pip installs instead of a mix of pip and conda, which causes issues when downloading packages for the GPU server
(Note when installing foundation-cancer-image-biomarker on the GPU serer I get the following error:
ERROR: project-lighter 0.0.2a19 has requirement loguru<0.7.0,>=0.6.0, but you'll have loguru 0.7.2 which is incompatible.
ERROR: project-lighter 0.0.2a19 has requirement pandas<2.0.0,>=1.5.3, but you'll have pandas 2.0.3 which is incompatible.)
* Added used packages into imageSegmentationMultipleSingleSlice.ipynb
* Added CLI command to run the notebook using jupyter nbconvert
 

Need to do [current]:
* Apply image transformations on these images
* Choose fixed image dimension for all images to put into model 
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

