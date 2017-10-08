# **Behavioral Cloning**

## Project 3

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `model.h5` containing a trained convolution neural network
* `drive.py` for driving the car in autonomous mode
* `writeup_report.md` this document you're reading now summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
$ python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the NVIDIA Autonomous Vehicle Team's CNN, preprocessed with a normalization layer and a cropping layer.

#### 2. Attempts to reduce overfitting in the model

Since the track data is largely a long turn to the left, I helped reduce 'left turn' overfitting by creating mirrored images which will now train the model on right turns. Also, I found just two epochs was enough before my validation loss was stable and before overfitting.

#### 3. Model parameter tuning

The only parameter that required much tuning was the number of epochs. As mentioned above, I found just two epochs was enough before my validation loss was stable.

#### 4. Appropriate training data

(I tried driving the simulator to create data but was very bad at it!)

I decided to use the given training data, all three cameras -- center, left, and right -- along with mirroring each camera to create even more and different data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a LeNet model at first to see how my model would do. This car was fine on the straighter parts of the track but ran into the water on the curve by bridge. It was a fun experienment but I knew the NVIDIA CNN was a more powerful model and I switched to that.

Along with using all three cameras, mirroring the cameras, I adjusted the epochs to until I found a validation loss that was acceptable but reduced the likelihood of overfitting.

#### 2. Final Model Architecture

The final model architecture (model.py lines 44-64) consisted of the NVIDIA convolution neural network with the normalization and cropping pre-processing layers.

The NVIDIA CNN consists of five convolutional layers followed by a flatten layer then three fully connected layers. This is a regression network whose last node predicts the steering angle (heading).

 Here is a visualization of the architecture from NVIDIA's blog.

![NVIDIA CNN](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

#### 3. Creation of the Training Set & Training Process

As noted earlier, I used the provided driving data for track 1. I did try driving and collecting data on my own (and it was fun) but it wasn't good data and it wasn't worth training on.

The provided data has roughly 8000 * 3 (center, left, right cameras) images. I doubled this by flipping each image, creating roughly ~48,000 training images and corresponding steering angels.

I set the training data set to randomly shuffle and put 20% of the data into a validation set. The ideal number of epochs was two for the NVIDIA CNN to get my validation loss to down around 0.035. I used mean square error as the error function to calculate the difference in predicted versus actual steering angle. I likewise used the 'adam' optimizer to manage the learning rate.


### Simulation

Watch my lap around the track with `run1.mp4`.
