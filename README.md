Current progress:
* added oversampling funciton to oversample the traininging set
* changed around undersampling to do it at the train/test split, no longer undersampling on test too
* need to always have drop_last=True for the training dataloader

Need to do [current]:
* try SGD and other params not covered
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images