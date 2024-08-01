Current progress:
* Changed the dataset to turn images into RGB
* No longer modify the model to accept single channel images
* No longer uses expand_dims() on the datasets to expliticitly turn them into single channel images
* testSamples31-7.py used older version, testSamples31-7-224x224.py uses newer version 
In generatePreprocessCombinations.ipynb:
* Now generates images in rectangles of 224x224 sizes regarless of the choice of scaledBoxes in preprocess().
** For scaled boxes of rectangular size, the padding is added to the centerXYofImage() so that the difference is of 1 pixel
** convertNdArrayToCV2Image() now turns images into 224x224 and is always used regardless of scaledBoxes passed.
** Generated new batches of images, with segmentsMultiple=12. This differs from the others by having the min number of slices of the presegmentation from 6, adding 4 more patients. In segmentsMultiple=13, we capped it at 7 minimum slices. From now on, end from 6 for the remainder of the cases. segmentsMultiple=13 will no longer be used, and so is not updated to reflect the changes. segmentsMultiple=1 reflects this. I elect to not care about the segment with only 3 slices for consistency and there is only the single case. The older ones are moved to the old/ inside of the same directory in local copies, but these .npy files are not added into the actual commits

Need to do [current]:
* Evaluate SMOTE performance
* Evaluate undersampling performance
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category
For model testing:
* experiment with batch size and learning rate (test with grid search)
* try SGD and different params for ADAM 
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images