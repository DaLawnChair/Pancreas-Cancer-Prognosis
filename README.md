Current progress:
* Added code to undersample the larger class in the generate k folds function. Note you should comment either the undersampler code or the SMOTE code
* Made the block where we first get the recistCriteria convert the labels into the versions we want.
* Added argumements to the generate k folds function as for some reason, requesting variables through print statements gives a undefined local variable error but calling it in stratifiedkFolds() doesn't
* Added argparse value for singleLargest, which adds confirmation that you want to do only the singleLargest datapoints. (ie with segmentsMultiple=1 and grouped2D=f). Added supporting code that uses it instead of grouped2d sometimes.

Need to do [current]:
* Evaluate SMOTE performance
* Evaluate undersampling performance
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category
For model testing:
* experiment with batch size and learning rate (test with grid search)
* try SGD and different params for ADAM 
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images