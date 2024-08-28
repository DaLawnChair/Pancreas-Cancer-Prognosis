Current progress:
* Generated new preprocessCombinations with recist criteria that groups complete response, partial response, and stable disease
* Added code to write to 3 seperate dataframes: singleLargest.xlsx, average.xlsx, and majority.xlsx, which hold the tests for the respective evaluation metrics.
* Added ability for the single generateKFoldsValidation() function to perform both average and majority evaluations (since they use the same model, they should do the same tests) Note that singleLargest theoretically may but shouldn't be done with the others due to grouped2D being true/false for each either. 
* Added 'multiVoting' for voting on both systems
* Added an append at the start to indicate model before setGridSearchParams() does the gridsearch evaluations

URGENT:
* Make InceptionV3Small2D work with singleLargest (ie singleLargest voting, 1 segment, grouped2D with False value). Previous version was using VGG16 (however I made sure in the segments beyond 1 were redone to use Inceptionv3)

Need to do [current]:
* try SGD and other params not covered
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images