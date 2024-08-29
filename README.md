Current progress:
* Fixed issue with models with batch normalization layers not accepting smaller data from segmentsMultiple=1 with singleLargest voting. This is because the dataset is too small with too few batches with ununiform sizes within the batches for the training dataloader. Made the training dataloader use drop the last batch for the training dataloader.
* Added test28-8.py, which is the segmentsMultiple=1 with singleLargest voting complement to test27-8.py
* Added NewRECISTGroupingTestsResults, a folder containing all tests performed regarding the new recist criteria and undersampled for segments=1,3,6,9,12 for singleLargest, average, and majority voting with the Formatted.xlsx prefix containing the formatted data.


Need to do [current]:
* try SGD and other params not covered
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images