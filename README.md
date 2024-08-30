Current progress:
* Changed behaviour of averageMultilabelMetricScores() to calculate the mean, sem, dictionary averages, and data; adding in the sem parameter.
* Updated the result of averageMultilabelMetricScores() to appendMetricsToXLSX() 
* updated Formatted.xlsx versions of NEWRECISTGroupingTestResults
* made the segmentsMultiple determine the voting system for running the gridsearch function, now can run all tests in 1 script.
* Added ability to easier generate transforms with adding Compose objects and getModelTransformation(), which houses the default transformations and can add in additional transformations

Need to do [current]:
* try SGD and other params not covered
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images