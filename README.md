# sail
S.A.I.L. (Semi Automated Image Labeling)

The goal of this system is to eliminate or greatly reduce the amount manual labeling\annotation needed to build deep learning classification models for 3D machine vision systems.

A number of different techniques are used to recursively train and test small LeNet style convolutional neural networks in Caffe deep learning framework. Other deep learning tools such as TensorFlow, Torch, and Theano would work very similar. Currently LMBD databases are created in Python from crops of coin images. Networks are trained with the stock Caffe command line and they are tested with a fork of the Caffe classification.cpp tool:
https://github.com/GemHunt/caffe/tree/master/examples/cpp_classification

**Installation:**
* Nvidia DIGITS needs to be installed on ubuntu 14.04 and tested. See:
* https://github.com/GemHunt/CoinSorter/blob/master/scripts/AWSCaffeDigetsBuild.md
* If you're using bash (on a Mac or GNU/Linux distro), add this line to your ~/.bashrc file:
    export PYTHONPATH=~/caffe/python:$PYTHONPATH

**Usage to Test:**
* Execution starts in rotational.py
* Download and untar sample images(13927 cent coin images, 180MB total):  http://www.gemhunt.com/cents.tar.gz
* Run build_init_rotational_networks() in rotational.py
* (for now) Pick the best model from cent-models/metadata/html

**Software Terms & Techniques:**
* Correlational Grouping: (Working Great!): I don’t know what to call this. It uses correlations within an image to group images or features/parts of images. Currently it’s using rotational correlations and they work awesome! I don’t see why other translations and scaling wouldn't work the same. See the readme: https://github.com/GemHunt/sail/tree/master/rotational
* Bootstrapping: This will the core of the system, but it’s not started yet. Recursively training manually labels results to provide the user with another set of results to label or just verify. This alone is a great speed up in manually labelling images. I have had great success with this in the past manually do this. It’s one of the most basic and easy self-supervision techniques.
* Graphing: (Code works good, but not currently used) Using graphing(as in mesh networks) to further associate label to other images
* Widening: (Code works good, but not currently used) Graphing & Bootstrapping combined

**Hardware Terms & Techniques:**
* Rotational Lighting Augmentation (This works awesome): Training many images of the same object all with different lighting angles to classify 3D objects. I discovered this after noticing coin images of the same design captured with side lighting group well within a 60 degree rotational window. Correlational Grouping and Rotational Lighting Augmentation are combined to build out “One Shot” feature classification. See: https://github.com/GemHunt/lighting-augmentation
* Camera Direction Augmentation: Training many images of the same object with different camera views.

**Current Tasks:**
* Back up a little Remove:     low_angle = 345      high_angle = 15
* Document the current pipeline

**Latter S.A.I.L. Tasks:**
* See the huge list of rotational tasks: https://github.com/GemHunt/sail/tree/master/rotational
* Put in a bootstrapping web app
* Use Flask to serve the image results
* Add a server process that can do the whole cycle by itself. It chooses what would the most helpful to work on next. It also chooses where to ask a human for help. So a human can intermittently help out. This is better than pure process with no human help because it can work with or without a human.
