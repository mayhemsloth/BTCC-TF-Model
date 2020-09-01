# BTCC-TF-Model

This repository was made to facilitate the creation of a TensorFlow model for the purpose of Between Two Castles Counter (an app made to automatically calculate the score of a castle in the board game Between Two Castles of Mad King Ludwig.) Link will be added to the app on Google Play Store when publically released. 

**Most of the progress and coding for this project has been moved to a Google Colab Jupyter Notebook that is not part of this repository, so most likely this repository will not be updated until the final model is produced**. 

# Method
Ultimately I followed the Python Lessons TensorFlow2.x YOLOv3 [tutorial here](https://pylessons.com/YOLOv3-TF2-introduction/) to do transfer learning for object detection using YOLOv3 model architecture. I created my own dataset, which was pictures of different configurations of the gamepieces used in Between Two Castles, and labeled the dataset using [labelImg](https://github.com/tzutalin/labelImg). After that, I attempted to train a model using transfer learning on my laptop, but it took too long to train on my 8-year-old laptop. I migrated to using Google Colab and training on there.

After that, training went much more smoothly. A Single Class model was created, and after demonstrating its ability to accurately identify the one tile it was trained for, I quickly moved to the All Class model (181 classes in total). It seemed that, with the dataset that I had available to me, the model's validation loss would stop decreasing after around 150-160 epochs. After that, we determined that, because the application for this NN would be used on phones, the YOLOv3-Tiny architecture would be better overall. The tutorial already had a Tiny model implementation, so that wasn't difficult to train. 

The model was then exported TensorFlow Lite format and given to my collaborators in charge of integrating the model into an app. However YOLOv3 outputs *all* the bounding boxes predictions. More code was needed to translate the YOLOv3 outputs to the actual best bounding box predictions, namely an algorithm with non-max suppression and intersection over union, but that was coded and implemented in the app. 

# Results
On August 31st a model was trained that was deemed "release-ready" quality. Therefore this project and repository will significantly slow down, because it is essentially finished. I successfully trained a YOLO architecture NN model for object detection that could directly identify the 181 unique tiles or cards in the game Between Two Casltes of Mad King Ludwig from just one picture of the final castle. I improved upon the tutorial code with the following:
* Implemented new color-affecting data augmentation techniques, specifically random versions of Saturation, Hue, Brightness, Contrast, and ultimately unused versions of Sharpness and Equalize
* Implemented new bounding box-affecting data augmentation techniques, specifically random versions of Rot90 and Warp. Additionally, unused version of CropToSize

# Takeaways
Here are the key takeaways for me from this project:
1. Establishing a full data augmentation + training pipeline should be done almost immediately. **Be able to train *some* ML model as soon as possible**. Because you can learn things from the model output to better hone what to work on next. Lot's of what I spent time on was dictated by what the previous iteration of the model was doing poorly.
2. If you are working with a small dataset, **robust data augmentation functions are absolutely critical**. Think about and write down all the ways that your positive data could be changed and still be identified and labeled as a positive by a human. After, write code to automatically transform your data (and labels when necessary) for *every method* on your list. My model's performance jumped **significantly** when I introduced the color changing augmentation functions, and then again when I introduced the Warp function.
3. **Building your own data set is time-consuming.** For this project I created literally the entire dataset by myself. I personally took all the pictures and labeled every one of them. I spaced it out over ~6 weeks so it wasn't too bad. It still took a significant amount of time and mindless label sessions. 
4. For object detection models, don't attempt a serious training run until you have **20 unique instances of each class**. Although the internet generally recommends 50+ instances of each class, I kind of ignored that recommendation as I was staring down the daunting task of 181 classes. I think if I had just taken 2 weeks, hunkered down and labeled a bunch more data initially, I might have been able to do everything faster overall. Here are some statistics on my final training set.
  * Number of classes: 181
  * Number of Average Unique Instances in Training Set per subcategory (total average = 22.3):
    * Royal Attendants: 100.0
    * Throne Rooms: 34.3
    * Bonus Cards: 19.6
    * Tiles: 20.0
