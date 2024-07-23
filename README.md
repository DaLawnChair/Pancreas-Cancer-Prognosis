Current progress:
* Fixed printing of evalDetailLine to the .xlsx file
* Added modularity for dropoutRate of model and easier time with testing different number of segments
* Added conditions to work on both notebooks and scripts with the argparse
* Fixed bug with argparse not giving the correct value for the bool type values
* Add check for when validation loss is nan, making it stop training
* Added storage and generation of .npy files for quicker loading of croppedSegmentsList, speeding it up from ~4 minutes on the computer to ~5 seconds

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