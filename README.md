Current progress:
* Changed f1 and recall to be a 3 element array of the f1 and recall of the evaluated test, respectively.
* Changed f1std and recallstd to include average metrics for each class, data also shows the multi-element array of the predictions
* Changing the datasets of grouped2D to behave differently:
** No longer produces (# of images)*(# of desired slices) instances, it now groups the images all under a patient id
** Added evaluateGroupVoting(), which performs majourity voting based on the grouping of patients, if required, otherwise it has the same functionality as evaluate()
** Changed around getting the answer set from the testingset in evaluateModelOnTestSet() 

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