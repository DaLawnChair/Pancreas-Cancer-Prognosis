#!/bin/bash
#SBATCH--job-name=2ClassPerformanceMajority
#SBATCH--qos=privileged
#SBATCH--partition=reserved
#SBATCH-c 8
#SBATCH--mem 16G
#SBATCH-o log.out
#SBATCH-e log.err
#SBATCH--mail-type=ALL
#SBATCH--mail-user=22jz10@queensu.ca
#SBATCH--time=0-2:0:0
#SBATCH--exclude=cac[029,030] 
source researchpip/bin/activate
module load StdEnv/2020	 
module load python/3.8.10 
python3 -u testSamples2-8.py -batchSize=8 -epochs=100 -lr=0.001 -evalDetailLine='2 class on majourity voting' -hasBackground=f -usesLargestBox=t -segmentsMultiple=12 -dropoutRate=0.2 -grouped2D=t -weightDecay=0.01 -modelChosen='Small2D' -votingSystem='majority'