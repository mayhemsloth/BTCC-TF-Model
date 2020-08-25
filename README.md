# BTCC-TF-Model

This repository was made to facilitate the creation of a TensorFlow model for the purpose of Between Two Castles Counter (an app made to automatically calculate the score of a castle in the board game Between Two Castles of Mad King Ludwig.) 

Ultimately I followed the Python Lessons TensorFlow2.x YOLOv3 [tutorial here](https://pylessons.com/YOLOv3-TF2-introduction/) to do transfer learning for object detection using YOLOv3 model architecture. I created my own dataset, which was pictures of different configurations of the gamepieces used in Between Two Castles, and labeled the dataset using [labelImg](https://github.com/tzutalin/labelImg). After that, I attempted to train a model using transfer learning on my laptop, but it took too long to train on my 8-year-old laptop. I migrated to using Google Colab and training on there.

After that, training went much more smoothly. A Single Class model was created, and after demonstrating its ability to accurately identify the one tile it was trained for, I quickly moved to the All Class model (182 classes in total). It seemed that, with the dataset that I had available to me, the model's validation loss would stop decreasing after around 150-160 epochs. After that, we determined that, because the application for this NN would be used on phones, the YOLOv3-Tiny architecture would be better overall. The tutorial already had a Tiny model implementation, so that wasn't difficult to train. 

The model was then exported TensorFlow Lite format and given to my collaborators in charge of integrating the model into an app. However YOLOv3 outputs *all* the bounding boxes predictions. More code was needed to translate the YOLOv3 outputs to the actual best bounding box predictions, namely an algorithm with non-max suppression and intersection over union, but that was coded and implemented in the app. 

**Most of the progress and coding for this project has been moved to a Google Colab Jupyter Notebook that is not part of this repository, so most likely this repository will not be updated until the final model is produced**. 
