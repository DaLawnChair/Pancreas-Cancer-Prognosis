Current progress:
* Added code to add a lr scheduler
* Made the model an inceptionv3 model, since the performance is much better thant eh original
* Reorganizing files
* Added all scripts that were used on the server for the tests


Need to do [current]:
* Refractor code
* Evaluate undersampling performance
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category
For model testing:
* experiment with batch size and learning rate (test with grid search)
* try SGD and different params for ADAM 
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images