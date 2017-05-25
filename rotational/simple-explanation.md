Rotational Correlation to Build Self-Supervised Deep Learning Models
 
This algorithm uses a standard classification convolutional neural network. You train it on the rotated images of the same object. I call this the seed. Then you test it on rotated images of other objects. The output is the angular difference between the two objects and the total probability.


**Benefits:**
* Can be unsupervised or only require a few images to be labeled. 
* You find the orientation in relation to the seed image
* Works as a rejection tool, as it’s better at giving an answer of “this is not the part”, when you have an input that has never been seen before. This is harder to do with straight classification where overfitting can easily cause false confidence on unseen input. 
 
**Usage:**
* A seed model is trained with 360 classes
* The seed model is then inferred(tested) with 360 rotations of the same image
* In all 360 tests: angle_differance = class result angle - test image angle 
* Grouping by the angle_differance you total the network output to get max_total_result
* The angle_differance with the max_total_result is the result for that seed
* Say you had 20 seed models and 300 coins to group:
* * The winning seed for each test image would be the max of the max_total_results for all 20 seeds
* * 20 seeds x 360 degrees x 300 coins = 2,160,000 images inferenced(tested)
 
**Notes:**
* I use this for coins, I can see it being used in all sorts of machine and robotic vision applications where the parts are very similar. 
* A more generic term would be graph based correlation. So this would include a positional correlation, which works the same as this, expect in X and Y. 
* When using this I noticed side lighting angles were grouping in 60 degree window so it took many images to make a model. This is how I came up with using many lighting angles with LED lights.

**Improvements**
* Try less than 360 classes
* Instead of using image of the whole coin, try a rotational windowing scheme

**Links:**
* https://github.com/GemHunt/sail has code, but I have to warn it’s overly complicated for demo code.
 
* Example groupings:
* http://www.GemHunt.com/designs/
* http://www.GemHunt.com/dates/
