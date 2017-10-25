## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration
* The size of training set is 3499
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
Here is a bar chart of the classes (before we create extras so that no classes are less than the mean):
![Class Bar chart][image1]

And a randomly printed sign:
![Random sign][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

The first steps I took were to normalize and grayscale all of the images, since color doesn't matter and adds extra data to analyze. I then iterated through each of the classes, and if there were less than the mean in that class, appended a random image rotated to the class. Then, all classes would have at least the previous mean.


#### 2. Describe what your final model architecture looks like 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolutional     	| 1x1 stride, same padding	                    |
| Activation            | Relu                                          |
| Pooling               | Max Pooling                                   |
| Convolutional         | Output 10x10x16                               |
| Activation            | 
| Pooling               | 
| Flatten               | 
| Fully Connected       | 
| Activation            | 
| Dropout               | Prevent over fitting                          |
| Fully Connected       |
| Activation            | 
| Dropout               |
| Fully Connected       | Output 43                                     |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I simply adjusted the hyperparameters to try to up the accuracy. I ended up changing the learning rate, drop probability (to prevent over fitting), and increased the batch size.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

The first thing I tried was leaving the default LeNet implementation, which wasn't getting me close. I implemented grayscale and normalization which improved it slightly,  and found I should even out the data in the classes since some had so many more images.

I also added a dropout layer to avoid overfitting. All of the parameters were then adjusted up and down in small increments to try to find an optimal solution.

My final model results were:
* validation set accuracy of 0.933
* test set accuracy of 0.915
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I used the `imagemagick` linux program to resize all of the images to 32x32. this results in some blurry and skewed photos, which lead to not great performance.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5kmh      		    | Speed limit (30)  [1] 							| 
| 2 way street     		| Right of way  [11]							    |
| Crosswalk				| Turn right ahead		[33]					    |
| Yield	      		    | End of no passing	[42]				 			|
| Stop Sign			    | Stop   [14]   							        |


The model was only able to guess 2 out of 5, giving an accuracy of 40%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

For the first image, the model was fairly certain it was a speed limit
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 2.4         			| Speed limit   									        | 
| 2.4     				| No vehicles 										    |
| 2.1					| Speed limit											    |
| 1.9	      			| Speed limit					 				            |
| 0.8				    | Speed limit      							            |


For the second image (inconclusive) ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3.1        			| Right of way   						        | 
| 2.8     				| General Caution 							    |
| 2 					| Roundabout			    		    	    |
| 1.4	      			| Pedestrians					 		        |
| 0.4				    | Traffic Signals      				            |

For the third (inconclusive) ...
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5.2         			| Turn right ahead   									        | 
| 0.2     				| Speed limit 										    |
| 0.1					| Go straight or left										    |
| 0	      			    | Double curve					 			            |
| 0				        | Dangerous curve to left      							            |

For the fourth (inconclusive) ...
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 10.7         			| End of no passing  (80kmh) 									        | 
| 7.6     				| End of speed limit 										    |
| 5.5					| End of no passing										    |
| 4.1	      			| Roundabout mandatory					 				        |
| 1.4				    | End of all speed/passing limits      							            |

For the fifth, fairly sure it is a stop sign, and is correct
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 23.0         			| Stop   									        | 
| 16.3     				| No entry 										    |
| 7.4					| Turn left ahead								    |
| 6.7	      			| Turn right ahead						            |
| 3.7				    | Yield      							            |



