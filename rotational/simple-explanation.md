Rotational Correlation to Build Self-Supervised Deep Learning Models
 
This work greats great for machine vision when the parts are very similar
 
Benefits:
Can be unsupervised or only require a few images to be labeled. 
You find the orientation in relation to the seed image
Works as a rejection tool, as it’s better at giving an answer of “this is not the part”, when you have an input that has never been seen before. This is harder to do with straight classification where overfitting can easily cause false confidence on unseen input. 
 
Notes:
A more generic term would be graph based correlation. So this would include a positional correlation, which works the same as this, expect in X and Y. 
 
Using this I noticed side lighting angles were grouping in 60 degree window so it took many images to make a model. This is how I came up with using many lighting angles with LED lights.
 
https://github.com/GemHunt/sail has code, but I have to warn it’s overly complicated for demo code.
 
Example groupings:
http://www.GemHunt.com/designs/
http://www.GemHunt.com/dates/
