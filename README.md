Current progress: 
* Moved back to 2D classification
* Changed batchsize from 1 to 8 and froze the layers of the resnet, seems to have fixed the issue of it only guessing 1
* Checking Accuracy, F1, and Recall of model. Currently being miscalculated as all 100%s

Need to do [current]:
* Find out why the accuracy, F1, and Recall of the model are always 100% despite testing accuracy being different
* Evaluate different types of evaluate metrics on the testing set
* Try different splits of the training set
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

