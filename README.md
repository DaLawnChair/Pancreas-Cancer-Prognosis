Current progress:
* Added code for using SMOTE inside of the folds to upsample the smaller classes.


Need to do [current]:
* Evaluate SMOTE performance
* Evaluate undersampling performance
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category
For model testing:
* experiment with batch size and learning rate (test with grid search)
* try SGD and different params for ADAM 
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images