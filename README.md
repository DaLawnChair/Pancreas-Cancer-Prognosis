Current progress:
* Converted numpy array methods into simplier sitk uses, refractoring code and speeding up times
* Added in conversion when 2 images are not aligned
* Adding in functionality for post segment and whole images

Need to do:
* Fix cases with post and whole not lining up, specifically when the z size of the whole segment is less than the post segments'
* Accept a 1D array for the case of CASE546
* View and fix case ids above CASE546
* Segment both post and pre with the whole image
* Combined these images into 1 numpy array
* Apply image transformations on these images
