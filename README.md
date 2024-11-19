Current progress:
* changed the spacing of the images to average spacing from 1mm^3


New:
* Now dedicates a testing set
* Reverted to 1x1x1 spacing for generatePreprocessCombinations
* getDataframes() now creates the sheets if they do not exist
* Needed to restructure __get_item__() to do this:
        image = (image * 255).astype(np.uint16)
        image = image[0]
        image = Image.fromarray(image)
        image = image.convert("RGB")
    instead of 
        image = Image.fromarray((image * 255).astype(np.uint16))
        image = image.convert("RGB")

* Set savemodel(saveModel=True) to save the model
Need to do [current]:
* try SGD and other params not covered
Later on optimizations
* look into meta-analysis - how and why the model inferences for particular images