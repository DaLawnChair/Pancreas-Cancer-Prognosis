Current progress: 
* Added tracker for groups (didn't implement the stratification yet)
* folders are now sorted, may have been causing issues depending on what computer ran the tests on
* Added parser to model 
* Added learning rate parameter
* Changed appendMetricsToXLSX() to only add values into a dataframe object that is open in the generateKFoldsValidation()
* Added method to add descriptor to of the experiment in addEvalDetailToModel()

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