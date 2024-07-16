Current progress: 
* Transforms: turned min-max normalization to standardization with mean dn std. Performing augmentations, and then normalization now
* Added important metrics to a dataframe for easy viewing of the final results after K fold crossvalidation
* Added new file to run the folder the images with background (THIS IS WHERE THE NEW CHANGES ARE)

Need to do [current]:
* Evaluate different types of evaluate metrics on the testing set
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category


ADD:
* put all important metrics in a csv file
* do validation set from the training set and see how it's accuracy is when training XXX
* implement early stopping XXX
* resize images to largest box size for all patients, and apply that box to all tumors to stop auto-rescaling 
* groupkfolds with classifcation on each slices, then do majourity voting on each prediciton to decide final classificaiton (gives way more data to work with and learn from)
* add guassian noise to the test Tranformation XXXX

For model testing:
* weight normalization layer, and view the different techniques (ie L2)
* experiment with batch size (test with grid search)
* try SGD and different params for ADAM 
* learning rate

Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images