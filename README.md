Current progress:
* Added new notebooks twoClassClassification.ipynb and twoClassClassificationMethods.ipynb. These are going to be the new working notebooks and house the using code and methods respectively
* made generatePreprocessCombinations.ipynb produce dictionaries with patient, images, and labels as dicationary enteries in the form of
{patient:{images: <images>, label:<label>}}
** Modified twoClassClassifications and twoClassClassificiationMethods accordingly for these changes
* Added models to the methods
* Removed args.parse, maybe will add again later.
* Temporarility removed SMOTE as it is difficult to get back the original images and their respective patient number after using it.
* Found the bug producing reproducibility errors, I think it was this after setting all other seeds and introducing a worker seed to the dataloaders: torch.backends.cudnn.benchmark = False # was True before
* remodeled testResultsNew.xlsx to hold data with better parameters and formating
* Now keeps track of precision and auc, and auc-roc curves
* Test folder now keeps track of actual used scripts, no longer need to worry much about losing versions now



Need to do [current]:
* Refractor code
* Evaluate undersampling performance
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category
For model testing:
* experiment with batch size and learning rate (test with grid search)
* try SGD and different params for ADAM 
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images