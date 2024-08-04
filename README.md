Current progress:
* Added a new notebook, imageSegmentation2Classes.ipynb, which is a modified version of imageSegmentationsWithBackground.ipynb where the models and data
are of type 0 or 1, where 0 represents progressive disease and stable disease, and 1 represents partial response and complete response
** Changed models and other stuff around to accomidate for this, including testing metrics and training/evaluations
* Changed behaviour of the getLargestSlice() function in generatePreprocessingCombinations.ipynb, also made the dataset generation into a function that can handle everything I want to run with it
** Added some stuff to the generatteXlSX notebook to see phesiablility of changing recist criteria thresholds for the patients -- it should not be done because the threshold would be in the positive for the change to put it into the underrepresented class. Most cases of stable disease are actually net improvement.


#!/bin/bash
#SBATCH --job-name=view 2 class performance on voting schemes
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH -c 16
#SBATCH --mem 16G
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=22jz10@queensu.ca
#SBATCH --time=0-2:0:0
#SBATCH --exclude=cac[029,030] 
The following is for running a Python script
source researchpip/bin/activate
module load StdEnv/2020	 
module load python/3.8.10 
python testSamples2-8.py -batchSize=8 -epochs=100 -lr=0.001 -evalDetailLine='2 class on majourity voting' -hasBackground=f -usesLargestBox=t -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -weightDecay=0.01 -modelChosen='Small2D' -votingSystem='majority' && testSamples2-8.py -batchSize=8 -epochs=100 -lr=0.001 -evalDetailLine='2 class on average voting' -hasBackground=f -usesLargestBox=t -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -weightDecay=0.01 -modelChosen='Small2D' -votingSystem='average'


After this has been created, press Ctrl+O to prompt saving your file. If you don’t want to change the name of your file, press the Enter key. To exit, press Ctrl+X. 
To submit this job to run, type the following command in terminal:
sbatch name_of_script.sh
You should get an output saying your job has been submitted with a specific job number. 
To see if you are in queue or if you’re code is currently running, enter the following command:
squeue -u hpcXXXX
To see the entire queue:
squeue



Need to do [current]:
* Evaluate SMOTE performance
* Evaluate undersampling performance
* Analyze and choose which features are important to the underlying goal, predicting progression through RECIST_PostNAC category
For model testing:
* experiment with batch size and learning rate (test with grid search)
* try SGD and different params for ADAM 
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images