Current progress: 
* Added a dataloader to the testing set
* Added more variables to describe the model more rigerously
* Added in a random seed of 0 for the transformations
* Cleaning imports and documentation
* Save and display training plots
* Turned the model into a class

Need to do [current]:
* Find out why model always predicts 1 in testing set, despite it getting 100% in every epoch in training set
* Why does the model get 100% on every epoch in the training set and have good 
* Evaluate different types of evaluate metrics on the testing set
* Try different splits of the training set
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category

Need to do [old]:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments' [This is due to the images themselves not lining up, not sure how to fix]
* Accept a 1D array for the case of CASE546 post-treatment
* View and fix case ids above CASE546

