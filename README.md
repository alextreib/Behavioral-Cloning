# Project: Behaviorial Cloning Project

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

---

Writeup
---

**Behaviorial Cloning Project**

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* drive.py for driving the car in autonomous mode
* model.h5 containing the weights for the neural network 
* model.json containing the model architecture of the neural network 
* preprocessing.py containing all the required image preprocessing steps
* utils.py containing small functions as reading the data and minor validation functions
* train.py containing the script to create and train the model (overall file with main function)
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As taken from the [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper the network consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully-connected layers. 

The first layer accepts an rgb image of size 66x200x3 and performs image normalization, resulting in features ranging from values -1.0 to 1.0, what is realized with a lambda expression.

After the CNN, the output is flattened and fed into fully-conncted-layers.
The last layer has as an output the steering angle.

The activation function used is the exponential linear unit (ELU), and an adaptive learning rate is used via the Adam optimizer
The weights of the network are trained to minimize the mean squared error

Finally, the used architecture contains about 27 million connections and 250 thousand parameters.


#### 2. Attempts to reduce overfitting in the model

In order to combat overfitting, the training data set is increased. First, the given training data set of Udacity is used, several rounds are recorded in order to increase the size of the dataset. This reduces the overfitting.

Moreover, a split for training/validation/test of the data is used, so the performance can be evaluated.
Through this, overfitting can also be detected and reduced with early stopping.

Furthermore, in the model architecture, several dropout are used, which is known for reducing the overfitting.

In the model, a L2 regularization reduces the overfitting further.

#### 3. Model parameter tuning

For hyperparameter tuning, an adam optimizer is used automatize it.
Most of the times, the epochs and samples per epochs were variated because they seem to have a huge effect on the performance.
The learning rate was modified, but is fine with the default value.

Due to the results in the mentioned paper, the model architecture was not modified largely.

#### 4. Appropriate training data

The 2/3 of the data set used for training is the one given by Udacity, but 1/3 is recorded manually.
First, a lot of self-recorded data was used, but the GIGO (Garbage In, garbage out) effect hits the performance.
Therefore, the training data was recorded with the mouse instead of the keyboard and the results were way better.

To enhance the given training data, some difficult parts of the road and some situations were the car is drive from the side of the lane to the middle of the lane (recovering) are added. This had very beneficial effects on the model. The difficult parts are passed way smoother.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My initial step was to try the known LeNet model from the traffic sign project. But the training with the Udadcity data was not so promising. 
Nevertheless, I added some preprocessing fucntion as the Lambda layer and Cropping layer. 
Well, it simply didn't work, so I did research and found the End to End learning paper by NVIDIA (see above).

I added the following image preprocessing steps to the model:
* Grayscaling
* Random brightness
* Cropping the image (cutting of unnecessary parts)
* Resizing for fitting into the model architecture

Furthermore, the probabilty that the image will be flipped is set to 0.5, so there will be no bias.

Moreover, the amount of low angles (below 0.1) is limited to 0.5 per batch because the data set is already biased with low angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (train.py lines 26-58) looks like this:

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 98, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 47, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 22, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 3, 20, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       dropout_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 1152)          0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      dropout_5[0][0]
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_6[0][0]
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_7[0][0]
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_8[0][0]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![This picture show how the center training data is accumulated][doc/center.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![From the left...][doc/moving_1.jpg]
![...slowly...][doc/moving_2.jpg]
![...to the middle][doc/moving_3.jpg]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images in order to reduce a potential bias.
For example, here is an image that has then been flipped:

![This picture represents how the dataset could be biased.][doc/moving_1.jpg]
![Through flipping the image, the bias is removed.][doc/moving_1_flipped.jpg]

After the collection process, around 15000 images are available for training, additionally to the Udacity training data.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 15-20 (trained with 20).
I used an adam optimizer so that manually training the learning rate wasn't necessary.




























### Data Collection

I used only the sample data provided by Udacity to train my model, although one could run the simulator in training mode to gather additional data. Udacity's driving simulator offers two different test tracks, and all sample data was collected from track 1 (the option on the left in the simulator's main screen). One of the optional challenges is to use only training data from track 1 and have the model navigate track 2 successfully; thus showing the model's ability to generalize. 
The data was collected by driving the car around the track in training mode, which records images from three separate cameras: left, center, and right. This is done so that we have data from car being on the left and right sides of the lane, and by adding an offset to the left and right steering angles we can train our model to correct back towards the center of the lane. Using all three of the cameras provides 24,108 samples on which to train.

![Left Image Example](example_assets/left_image_example.jpeg) ![Center Image Example](example_assets/center_image_example.jpeg) ![Right Image Example](example_assets/right_image_example.jpeg)

### Data Augmentation and Preprocessing

In an effort to reduce overfitting and increase my model's ability to generalize for driving on unseen roads, I artificially increased my dataset using a couple of proven image augmentation techniques. One method I used was randomly adjusting the brightness of the images. This is done by converting the image to HSV color space, scaling up or down the V channel by a random factor, and converting the image back to RGB. Another technique I used was flipping the image about the vertical axis and negating the steering angle. The idea here is to attempt to get an equal number of left and right steering angles in the training data to reduce any possible bias of left turns vs right turns or vice versa.
The original size of each image is 160x320. I crop the top 40 pixels and the bottom 20 pixels from each image in order to remove any noise from the sky or trees in the top of the images and the car's hood from the bottom of the image. This results in image sizes of 100x320, which I then resize to 66x200, which is the input size of the neural network.

![Left Image Processed](example_assets/left_image_processed.jpeg) ![Center Image Processed](example_assets/center_image_processed.jpeg) ![Right Image Processed](example_assets/right_image_processed.jpeg)

### Network Architecture

Here, a 


The network consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully-connected layers. The first layer accepts an rgb image of size 66x200x3 and performs image normalization, resulting in features ranging from values -1.0 to 1.0. The first convolutional layer accepts input of 3&#64;66x200, has a filter of size 5x5 and stride of 2x2, resulting in output of 24&#64;31x98. The second convolutional layer then applies a filter of size 5x5 with stride of 2x2, resulting in and output of 36&#64;14x47. The third convolutional layer then applies a filter of size 5x5 with stride of 2x2, resulting in output of 48&#64;5x22. The fourth convolutional layer applies a filter of 3x3 with stride of 1x1 (no stride), resulting in output of 64&#64;3x20. The fifth and final convolutional layer then applies a filter of 3x3 with no stride, resulting in output of 64&#64;1x18. The output is then flattened to produce a layer of 1164 neurons. The first fully-connected layer then results in output of 100 neurons, followed by 50 neurons, 10 neurons, and finally produces an output representing the steering angle. A detailed image and summary of the network can be found below. I use dropout after each layer with drop probabilities ranging from 0.1 after the first convolutional layer to 0.5 after the final fully-connected layer. In addition, I use l2 weight regularization of 0.001. The activation function used is the exponential linear unit (ELU), and an adaptive learning rate is used via the Adam optimizer. The weights of the network are trained to minimize the mean squared error between the steering command output by the network and the steering angles of the images from the sample dataset. This architecture contains about 27 million connections and 252 thousand parameters.

![Network Architecture](example_assets/network_architecture.png)  ![Model Summary](example_assets/model_summary.png)

### Training Details

The data provided by Udacity contains only steering angles for the center image, so in order to effectively use the left and right images during training, I added an offset of .275 to the left images and subtracted .275 from the right images. This is because an angle of 0 corresponds with the car going straight, left turns are negative, and right turns are positive. The angle of .275 was found by trial and error, since the distance between the cameras was not given. I also limited the number of angles less than the absolute value of 0.1 to a maximum of 50% of any given batch. This is done to prevent a bias towards the car driving straight, since the provided data had a high proportion of small angles. Also, images are flipped with their steering angle negated with probability of .5 in order to achieve a balanced distribution of left and right turns in the training data. Also in regards to balanced data: images are randomly selected by index, then by left, center, or right in efforts to create a balanced dataset.
Training is done via a keras fit generator, which constantly feeds the model data for a specified number of epochs and samples per epoch. The generator is used so that the training data is only loaded into memory in batches, as all of the training data cannot be loaded at once due to memory contraints. My model was trained for 28 epochs with 24,000 samples per epoch and batch sizes of 64. The samples per epoch being set at 24,000 is due to the fact the an epoch is generally defined as a complete pass through the entire dataset, which was just over 24,000. I arrived at 28 epochs through complete trial and error. I originally set epochs to 40, but the model seemed to be overfitting at that number. I split my data into training and validation at a rate of 90/10, and my validation error was lowest at aroung 0.03 after 28 epochs.

### Evaluation

The model is able to successfully navigate both track 1 and track 2 without much trouble. There are a couple areas of track 1 where the steering isn't very smooth, specifically around the first sharp turn. Surprisingly, the model performs better on track 2, from which it hasn't seen any sample data. The drive on track 2 is flawless and the car almost always stays in the center of the track. Overall, I am very happy with the outcome of this project and I look forward to one day testing some of my models on a real car. 

You can view full vidoes of my test runs here: [Track 1](https://youtu.be/rA6xbC0J1aQ) [Track 2](https://youtu.be/ajTGP91XADg)





