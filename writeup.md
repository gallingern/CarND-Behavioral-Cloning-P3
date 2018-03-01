# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/mse_2_epoch.png "2 Epoch Error"
[image2]: ./examples/mse_3_epoch.png "3 Epoch Error"
[image3]: ./examples/cnn-architecture.png "CNN Architecture"
[image4]: ./examples/driving_image.jpg "Driving Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based off of the NVIDIA neural network architecture for self driving cars described here:

https://devblogs.nvidia.com/deep-learning-self-driving-cars/

It is consists of five convolutional layers with ReLU activation, followed by three fully connected layers proceeded by 50% dropout to prevent overfitting.  The data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.   Also, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a center lane driving in the provided dataset for my training.  I planned to augment this with recovery data from the left and right sides of the road, but this was not necessary with my network architecture.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to employ a very basic network just to test that the pipeline and data processing were working correctly.  I did basic augmentation of the data by flipping the image from the center camera and inverting the steering angle.  Also, I added in the images from left and right cameras with a steering correction offset, and cropped the non useful top and bottom margin.  The result was a model with very poor jerky performance, as expected.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I then employed the NVIDIA architecture above.  I added dropout before the fully connected layers to prevent overfitting, and used the adam optimizer for three epochs.  The plot of the error loss is below.

![3 Epoch Error][image2]

I could see that the validation error was increasing by the third epoch, so I changed the epochs to two and got the resulting error.

![2 Epoch Error][image1]

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road, and stays almost perfectly centered the entire time.  I credit the sophistication of the NVIDIA architecture and the ease with which Keras enables constructing neural networks.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer					| Description									|
|:---------------------:|:---------------------------------------------:|
| Input					| 160x320x3 image								|
| Cropping				| ((70,25),(0,0))								|
| Convolution			| 5x5 filter=24 activation=relu					|
| Convolution			| 5x5 filter=36 activation=relu					|
| Convolution			| 5x5 filter=48 activation=relu					|
| Convolution			| 3x3 filter=64 activation=relu					|
| Convolution			| 3x3 filter=64 activation=relu					|
| Flatten				|												|
| Dropout				| 50%											|
| Dense					| 100											|
| Dropout				| 50%											|
| Dense					| 50											|
| Dropout				| 50%											|
| Dense					| 1												|

Here is a visualization of the architecture courtesy of NVIDIA.

![CNN Architecture][image3]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the provided dataset of the car on track one using center lane driving. Here is an example image of center lane driving:

![Driving Image][image4]

To augment the data sat, I also flipped images and angles, as well as using images from the right and left cameras with a steering offset.  I planned to recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover, but the driving was so smooth after initial training that this was unnecessary.

 I trained the model with 24,108 data points. I then preprocessed this data by cropping out the top and bottom margin of the image that didn't contain useful road image data.


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the error plots in the section above.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
