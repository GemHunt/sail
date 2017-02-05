# sail
S.A.I.L. (Semi Automated Image Labeling)

The goal of this system is to eliminate or greatly reduce the amount manual labeling\annotation needed to build deep learning models with very similar images and 3D profiles. A number of different techniques are used to recursively train and test small LeNet style convolutional neural networks in Caffe deep learning framework. Other deep learning tools such as TensorFlow, Torch, and Theano would work very similar. Currently LMBD databases are created in Python. Networks are trained with the stock Caffe command line and they are tested with a fork of the Caffe classification.cpp tool:
https://github.com/GemHunt/caffe/tree/master/examples/cpp_classification

**Terms & Techniques:**
* Bootstrapping: (Web App Not Started) Recursively training manually labels results to provide the user with another set of results to label or just verify. This alone is a great speed up in manually labelling images. I have had great success with this in the past manually do this. It’s one of the most basic and easy self-supervision techniques.
* Correlational Grouping: (Working Great!): I don’t know what to call this. It uses correlations within an image to group images or features/parts of images. Currently it it using rotational correlations and they work awesome! I don’t see why other translations and scaling wouldn't work the same.
* Rotational Lighting Augmentation (This works awesome): Training many images of the same object all with different lighting angles to classify 3D objects. I discovered this after noticing coin images of the same design captured with side lighting group well within a 60 degree rotational window. Correlational Grouping and Rotational Lighting Augmentation are combined to build out “One Shot” 3D feature classification.
* Camera Direction Augmentation: Training many images of the same object with different camera views.
* Graphing: (Code works good, but not used) Using graphing(as in mesh networks) to further associate label to other images
* Widening: (Code works good, but not used) Graphing & Bootstrapping combined

**Current Tasks:**
* Back up a little Remove:     low_angle = 345      high_angle = 15
* Document the current pipeline
* Generically add X translations,Y translations, and off center options to the pipeline. Store the offsets by using the image_id.
* Off Center Secondary Networks (Dates): (In a quick & dirty test I did, this works great!) Once a network is built for a design and the angle is known the same processes can be repleted for any image crop off center. At first this is for grouping dates, but it can be used for anything. This is a double check to coin designs.


**Latter Tasks:**
* Better ensembling! These correlations are not summed very well at all. I need to be able to add the predictions better.
* Does rotational lighting augmentation really work that great? Try captures with a cheap USB microscope using just the stock top lighting.
* Put in a bootstrapping web app with limited manual training
* Use Flask to serve the image results
* Add a server process that can do the whole cycle by itself. It chooses what would the most helpful to work on next. It also chooses where to ask a human for help. So a human can intermittently help out. This is better than pure process with no human help because it can work with or without a human.

