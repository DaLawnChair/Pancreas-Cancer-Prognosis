Current progress: 
* Added plots to the training histories
* when getting data, the data will be normalized, then augmented, and the normalized again, to be easier to view the differences in normalization
* Trippled the size of the model with adding 2 layers. Made the output layer to to a softmax function rather than the softmax function values go to the output layer
* Added some data augmentations to try out, they scale linearly with in intensity
* Made excel sheet include ending epochs, not just the singular one
* Made learning rate of the  model adjustable

Need to do [current]:
* Evaluate different types of evaluate metrics on the testing set
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category


ADD:
* resize images to largest box size for all patients, and apply that box to all tumors to stop auto-rescaling 
* groupkfolds with classifcation on each slices, then do majourity voting on each prediciton to decide final classificaiton (gives way more data to work with and learn from)

For model testing:
* weight normalization layer, and view the different techniques (ie L2)
* experiment with batch size (test with grid search)
* try SGD and different params for ADAM 
* learning rate

Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images