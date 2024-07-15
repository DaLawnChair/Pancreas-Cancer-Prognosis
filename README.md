Current progress: 
* Updated the model to have batchnormalization and dropout
* Added in methods to make transform.compose objects to test different parameters

Need to do [current]:
* Look into why the performance of each fold is the same, but also why between different scenarios are also the same (perhaps random rotation doesn't do much)
* Evaluate different types of evaluate metrics on the testing set

* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

