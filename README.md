Current progress:
* Changed transforms tested to non-normalized values, no longer normalizing image, applying augment, then normalizing again
* Updated weights of image
* Set the 0 transform to None instead of having each transform's value as very low
* Added in a default training_transform convertDataToLoaders(), made the dataloader for training have shuffle=True
* Made a smaller model to evaluate differences easier between models, renamed the old resnet50classificaiton model to large2d
* Updated to newer weights, much better performance (0.27 vs 0.49 for average accuracy for the small2D model, not that noticiable changes for large2d model)
* No longer doing a train, val, test set. Now doing train and test set where val=test


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