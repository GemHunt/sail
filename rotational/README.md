# rotational
Self-Supervised Rotational Clustering Networks

A fast tool to group coin designs and determine orientation angles with no manual labeling

**Installation:**
* Nvidia DIGITS needs to be installed on ubuntu 14.04 and tested. See:
* https://github.com/GemHunt/CoinSorter/blob/master/scripts/AWSCaffeDigetsBuild.md
* If you're using bash (on a Mac or GNU/Linux distro), add this line to your ~/.bashrc file: 
    export PYTHONPATH=~/caffe/python:$PYTHONPATH

**Usage:**
* Execution starts in rotational.py
* Download and untar sample images(13927 cent coin images, 180MB total):
* http://www.gemhunt.com/cents.tar.gz
* Run build_init_rotational_networks() in rotational.py
* (for now) Pick the best model from cent-models/metadata/html

**Tasks:**
* Document how this is done
* Start over and go for heads & tails again
* Fix: something in widen_model deleted all metadata?
* Huge: Generically add X translations,Y translations, and off center options to the pipeline. Store the offsets by using the image_id so the translations to be recursively applied. Generically define the range of the translation, as in what is this training and testing?
* What happens when the rotation network is tested at courser angles? Say 10 degrees instead of 1 degree?
* Off Center Secondary Networks (Dates): (In a quick & dirty test I did, this works great!) Once a network is built for a design and the angle is known the same processes can be repleted for any image crop off center. At first this is for grouping dates, but it can be used for anything. This is a double check to coin designs.
* Add results options:
* ---Show only one image per coin using the average correct results for all the same coin images.
* ---View least correlated coins in all test results.
* ---View least correlated images in all test results.
* Fix or acknowledge???: The current error checking(multi_point_error_test_image_ids) fails when seeds are too closely correlated.


**Later Tasks:**
* Does rotational lighting augmentation really work that great? Try captures with a cheap USB microscope using just the stock top lighting.
* Only resize once.
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

**Circular Lighting:**
* This is a big one as it can build a full class from a single scan. Light every frame with a different LED from a different angle as the part is moving under the camera. Use WS2812 LED Strips to get 8-30 different shadows. Use the 3 color channels in the LED to scan 3 angles at once! Also I could take all lights on pictures, and pictures with lights different heights. Not fancy, just 3 light strips loops. One loop horizontal, one vertical, and one under for separate backlighting shots. They don’t have to be mounted perfectly. The LEDs can switch every frame. You check the every n frames to see if the times is correct. The belt would always be moving. So both the camera and the lighting would be different models. I could make the models every image for itself then I could ensemble them. This is cool because it will show when the models are bad. The reason for this is also because this setup is going to work not just for coins, but other part types as well with no changes.

**Back lighting**:
* Scan screws without an image at all:1 bit backlighting from 8 different angles would the same as 256 gray but really it should be blob input instead of an image.
* You could have 25 different backlighting channels with cameras on top and bottom with the frosted belt. This is sloppy 3d scanning without doing the math.
* This will work for sure with the screws not touching, but can be it work with the screws touching?
* An old flat panel display can be both the backlight and it can light up around the issue screws.

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
* Try adding images
* Try rewriting lmdbs, while removing image_ids.
* Try adding and subtracting images from lmdbs.
* C++ GPU driven rotate tool for LMDBs? I bet this could be 100 times on the GPU & C++ and not that hard to do.
