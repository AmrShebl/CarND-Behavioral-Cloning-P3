# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/image1.jpg "Sample Image"
[image6]: ./examples/before_flip.jpg "Normal Image"
[image7]: ./examples/after_flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it is very much straight forward and self explanatory.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia network architecture introduced in the course.
The first layer in the network is a lambda layer used as a preprocessing layer to normalize the images and center their mean around zero.
The second layer is a cropping layer used to remove the top 70 rows of pixels that are mainly the sky and the trees as well as the bottom 25 rows of pixels which are the hood of the car.
This is then followed by three convolutional layers with a kernel size of 5\*5 and a stride length of 2. Each layer was followed by a relu activation layer.
This is then followed by a couple of more convolution layers with a kernel size of 3\*3 and a stide length of 1. The activation function for these layers was also a relu.
This is then followed by four fully connected layers in addition to the output layer that specifies the steering angle.

#### 2. Attempts to reduce overfitting in the model

The model was trained on a large amount of data to ensure that we don't have overfitting. The loss curves of the training and the validation sets were ploted and they showed the same trends. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 50).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I had a lap for clockwise and another for counter clockwise. I also had extra data for the bridge as it had different texture than the rest of the track. My main focus while collecting the data was to keep the steering angle continuous without sudden uprupt changes especially at curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the Nvidia's architecture introduced in the course and it was good enough. I didn't need to make any further changes to that. For me, the challenge was in getting the appropriate training data.

I run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I used the left and right cameras with a correction of the steering angle. This increased the data by a factor of three. I then flipped all the images along with their corresponding steering angles. This increased the data by an extra factor of two.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 37-49) consisted of a convolution neural network with the following layers and layer sizes:
1- A preprossing layer that normalizes the images and centers the mean around zero.
2- A cropping layer that removes 70 pixel rows from the top, that represent the sky and the trees, as well as 25 pixel rows from the bottom, which represent the hood of the car.
3- This was followed by three convolutional layers each with a kernel size of 5\*5, a stride length of 2\*2 and valid padding. A Relu activation was used after each layer. The number of filters was 24, 36, and 48 respectively.
4- An additional couple of convolutional layers were then added. They had a kernel size of 3\*3 and a stride length of 1 each. Relu activation was also used after each layer. The number of filters in both layers was 64.
5- This was followed by four fully connected layers with no activation. Their sizes were 1164, 100, 50, and 10 respectively.
6- The final layer is the output layer with a single node.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one that is going around the track in the clockwise direction and the other is in the counter-clockwise direction. The main focus in the two tracks is to have smooth continuous steering angles especially in curves and reducting the sudden changes as much as I can. Here is an example image:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would balance any bias towards either direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

The left and the right images were also used with a corrected steering angle. the correction was 0.2.

Data preprocessing is part of the network architecture discussed before. It is done basically with a lambda layer.

The data is shuffled in keras by setting the shuffling flag to true. I put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the graph of the mean square error vs the epoch number. I used an adam optimizer so that manually training the learning rate wasn't necessary.
