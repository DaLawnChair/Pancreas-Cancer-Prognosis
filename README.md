Current progress:
* Added stratifiedGroupKFolds as a possible arrangement for the model, it uses all slices from the 3D scans and treats them as individual images under a caseID to increase datapoints and prevent data leaking
* Added check for forward() where the batchnorm layers are not calculated if the batch size is 1 as this can give issues
* Updated the 3d model
* Updated the evalDetailLine to have the entire argument line
* A lot of data got deleted from my test cases :sadface:
* Reformated the xlsx to be easier to read and have more information

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