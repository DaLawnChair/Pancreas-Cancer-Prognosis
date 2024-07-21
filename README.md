Current progress: 
* No longer taking the transpose of the image in the preprocessing stage, now works with the proper form of [slice, width, height]
* Added implementation with adding background/no background and with global largest box or by fit-and-scale individual tumors
* Changed the view of the displayCroppedSegmentations() to display on grayscale of the windows. Implementing the new changes makes the background not completely black for whatever reason. I have no idea why it now doesn't auto-black everything outside of the range of the segment
* Preprocess() no longer has the variables for the upper and lower bounds of the window
* More testing needed to ensure that the preprocessing performs properly on all segments

Need to do [current]:
* Evaluate different types of evaluate metrics on the testing set
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category


ADD:
* resize images to largest box size for all patients, and apply that box to all tumors to stop auto-rescaling 
* groupkfolds with classifcation on each slices, then do majourity voting on each prediciton to decide final classificaiton (gives way more data to work with and learn from)

For model testing:
* weight normalization layer, and view the different techniques (ie L2 and AdamW)
* experiment with batch size (test with grid search)
* try SGD and different params for ADAM 
* learning rate

Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images