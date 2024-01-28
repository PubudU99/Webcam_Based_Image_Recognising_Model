# Webcam_Based_Image_Recognising_Model
This repository contains a computer vision model for image classification implemented using Keras with a TensorFlow backend
The model is trained for recognizing objects in images and is accompanied by a script for real-time webcam-based inference. The image recognition model is loaded from a pre-trained Keras model file (keras_Model.h5), and the class labels used for prediction are stored in a text file (labels.txt)

# About the ML and the Teachable Machine
I have used a Online Model Training Site called Teachable Machine for this.In that website you can train image recognition models easily without using the keras On your own.This actually train a neiral network for the model.We can add classes easily, For feeding we can choose the webcam and It is easy for adding new classes to the model.I have used 4 classes Namely SCENT,No Object, Colonge and Purse.

There are two files in the trined model,Labels.txt and the keras_model.h5.The labels contained the class names,For an example if the trained model is recognised an object it is shown as one of the classes in the labels.txt.The data of the images are saved in the model.

# Getting images from webcam continuosly
I have implemented a python code for getting images from the webcam and send the images for the model continuously at 2 sec time intervals.

# Image Cropping
The webcam image was quite bit large for the model,(ex: i have used a logitech webcam and the image size is bit large,so I have used image cropping in python for cropping the image.
i have used PIL library in the python for cropping the image.

# Image resizing
I have trained the model for 224*224 image size,So i have to resize the corpped image for the model,I n the combined.py You can find how i did it.

# Image recognision
After getting the data freom the webcam it is feeded for the model and then model identifies the class which it is belong the most then it returns the class name and the confidence score.

# Changing the repo according to your Will
Put your trained model and the labels files in the cloned folder after deleting the earlier(my model and the labels file)files.

# How to run the code
You can simply type python -m combined.py or click the play button in vs code.
