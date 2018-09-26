# Hand Gesture Recognition Using Background Ellimination and Convolution Neural Network

## About the Project

This is a simple application of Convolution Neural Networks combined with background ellimination to detect different hand gestures. A background ellimination algorithm extracts the hand image from webcam and uses it to train as well predict the type of gesture that is. More information about the algorithm can be found below.

## Requirements

* Python3
* Tensorflow
* TfLearn
* Opencv (cv2) for python3
* Numpy
* Pillow (PIL)
* Imutils

## File Description

[PalmTracker.py](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network/blob/master/PalmTracker.py) : Run this file to generate custom datasets. Go into the file and change the name of the directory and make other appropriate changes.

[ResizeImages.py](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network/blob/master/ResizeImages.py) : Run this file after PalmTracker.py in order to resize the images so that it can be fed into the Convolution Neural Network designed using tensorflow. The network accepts 89 x 100 dimensional image.

[ModelTrainer.ipynb](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network/blob/master/ModelTrainer.ipynb) : This is the model trainer file. Run this file if you want to retrain the model using your custom dataset.

[ContinuousGesturePredictor.py](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network/blob/master/ContinuousGesturePredictor.py) : Running this file opens up your webcam and takes continuous frames of your hand image and then predicts the class of your hand gesture in realtime.

## Some key architectural insights into the project

### Background Ellimination Algorithm

I have used opencv for taking a running average of the background for 30 frames and then use that running average to detect the hand that has to be introduced after the background has been properly recognized.

I had found a very useful article on foreground mask by [Gogul09](https://github.com/Gogul09) and i have pretty much used his code for background ellimination with a few changes in order to suit my cause.

He has written an awesome article on the problem and you can read it up [here](https://gogul09.github.io/software/hand-gesture-recognition-p1).

### The Deep Convolution Neural Network

The network contains **7** hidden convolution layers with **Relu** as the activation function and **1** Fully connected layer.

The network is trained across **50** iterations with a batch size of **64**.

I kind of saw that 50 iterations kind of trains the model well and there is no increase in validation accuracy along the lines so that should be enough.

The model achieves an accuracy of **96.6%** on the validation dataset.

The ratio of training set to validation set is **1000 : 100**.

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)
