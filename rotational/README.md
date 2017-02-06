# rotational
Self-Supervised Correlational Clustering Networks
Feature detection & localization. Anomaly detection.
(I don’t know what to call it! I keep changing the name. Regardless, it works great!)

A fast tool to group coin designs and determine orientation angles with no manual labeling

**Tasks:**
* Document how this is done
* Add in a train_id or something to manage test runs
* Move hard coded vars to config file and manage with a train_id
* Huge: Generically add X translations,Y translations, and off center options to the pipeline. Store the offsets by using the image_id so the translations to be recursively applied. Generically define the range of the translation, as in what is this training and testing?
* What happens when the rotation network is tested at courser angles? Say 10 degrees instead of 1 degree?
* Off Center Secondary Networks (Dates): (In a quick & dirty test I did, this works great!) Once a network is built for a design and the angle is known the same processes can be repleted for any image crop off center. At first this is for grouping dates, but it can be used for anything. This is a double check to coin designs.
* Add results options:
* ---Show only one image per coin using the average correct results for all the same coin images.
* ---View least correlated coins in all test results.
* ---View least correlated images in all test results.
* Fix or acknowledge???: The current error checking(multi_point_error_test_image_ids) fails when seeds are too closely correlated.
* Change how Get_multi_point_error_test_image_ids works:
* ---this might be a weak spot. Maybe scale needs to be broken out to separate similar networks
* ---There should be a per seed output of class and angle.
* ---Average the angle because all the angle should be the same
* ---Take all for a seed if more than 50% are errors.
* ---This is important because these 1-4 groups of coins that get really need to be part of the next seed
* Make a sample composite image of what is going to the lmdb file, or write/find a lmdb viewer) It’s easy to lose debugging time due to bad lmdb creation.

**Later Tasks:**
* Go beyond a fixed learning rate during training. Dropping the base_lr at the end of training would help.
* Create_single_lmdb needs to guess the amount of training data better. Up to 2800 samples is silly at times. Maybe this needs determine the epic count as well.
* Does rotational lighting augmentation really work that great? Try captures with a cheap USB microscope using just the stock top lighting.
* Use crop_size again.
* Remove 'pkrush' from the repo
* does remove_widened_seeds need to be an option?
* in save_widened_seeds: This should be widened_images
    (images that correspond to a widened seed)
* create images should not be in read_all_results

**Set up png's for serving:**
* Build a html file for the png’s
* Tar the png folder.
* Use a less bold font?
* sort by size?

**Just have 2 test groups(dump widening):**
* 0:  -29 to 29
* 1: All


**Used & Works Great:**
* Use best exclusive test result models with non-exclusive test images
* Expand model from a 60 degree window to 360 using a shortest path graph function

**Latter Improvements:**
* Speed: Different GPUs, multi threading, move more to C++
* Reduce work: For example about using 5 degree steps instead of 1?
* Better ensembling! These correlations are not summed very well at all. I need to be able to add the predictions better.
* Ensemble multiple models at different cropping(and remapped ring-crop) resolutions
* Ensemble in 2nd level models from off center cropping such as dates
* Getting dates might help the first level models. If this off center tool scores low the 1st model results might be in question.
* Widening (was useful and tried already, but graphing short paths was somewhat better)
* Merging many working models (might be the same as “graphing short paths”)
* Add more images of rare classes. (More scanning: looping?)
* Correlating many shots: same coin, same camera
* Correlating many shots: same coin, different camera
* 2 camera(top & bottom) correlation
* Better network tech (vgg, resnet, dropout, more layers, etc)
* Test different network sizes: 28x28 256 gray,stock lenet used. Image size and number of outputs can be changed to increase accuracy at the cost of speed)
* Camera calibration
* Test more with basic cnn parameters(epoch #, better transfer weights, learning rate, etc)
* Match checking from siamese networks from other coin networks already build
* Center alignment(x,y) in the same way orientation is being done
* Mean subtraction per camera & lighting setup
* Use graphing tools more for clustering or bridges / cut-sets to find issues
* Improve inference results: Don’t just pick the top angle in the, but do a running average first. A lot of the results are very close around the max.
* Use traditional CV features: SIFT, HOG, Etc
* Use really simple features such as some designs are larger png file sizes
* Research: Use known self-supervision tools (papers)
* Research: Talk with others about what works best
* Hardware Reject/Online Training: In practice I could reject and rescan parts that don’t have a class.

**Angle error reduction with short path graphs:**
* Start with a tiny network:
* Pick 2 points at random
* Average the angles for the 20 shortest paths.
* Add up the total ABS(Errors) each time
* Change each edge by .1 Degree. (Not percentage so the angles can flip past zero)
* Repeat 100 times
* Print the total error
* Repeat this loop
* Does the total error start dropping?

**LMDB Improvements:**
* Try editing pre existing lmdbs: as adding and subtracting images.
* create images using C++ GPU instead of python. 100x speed up.
