Current progress: 
* Added weighted sampler for the training set, balancing the classes (perhaps not working properly, need to reevaluate)
* Added a ResNet50 sampler model which will be used to evaluate my data agumentaton
* Added measures to save model training history, weights, scripts and notebooks
* Added method to read in history from pickle file

Need to do [current]:
* Add in testing validation training set
* Evaluate different types of evaluate metrics on the testing set
* Try different splits of the training set
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

