# Deep-Feed-Forward-Neural-Network-Tensorflow

## Overview

In this project our knowledge about Deep Neural Networks is tested as we are asked to implement any three types of deep neural networks using Keras or Tensorflow libraries. We are required to implement the experiments on our classification dataset that we have used in the past assignments. I will be using the Tensorflow library and deploying the model on my adult income dataset from UCI Repository. As we are asked to implement three types of networks. I will be implementing deep feed forward network using different combinations of hidden layers to understand affect of different combinations of hidden layers on the performance of the model. I will be implementing models with increasing number of nodes, decreasing number of nodes and then alternating number of nodes. I will be using the following set of hidden layers = {10,20,30,40,50,60}. Also, in our third assignment of Neural Networks, we had implemented Neural Network for our classification dataset, we can use the same class provided by the professor and convert it into a deep neural network by increasing the number of hidden layers. But, for now to experiment with a new code, I have implemented Deep Feed Forwards network using Tensorflow library.

## Data

### Introduction of data and source
The dataset that I have selected to visualize represents the income census information of adult individuals and I have selected this income census dataset from the UCI repository(UCI Machine Learning Repository: Adult Data Set). This dataset has been traced from 1994 census database and comprises of information about age groups, work-classes, level of education, country, occupations, etc. It comprises of the following labels:
* age
* workclass
* fnlwgt
* education
* education_num
* marital_status
* occupation
* relationship
* race
* sex
* capital_gain
* capital_loss
* hours_per_week
* native_country
* income

Datasets like this one is widely used for building prediction models about future census information. This dataset houses information of more than 30,000 people. Such datasets can be really helpful in predicting future census information if studied well and can be a good resource.

### Preprocessing
In this section, I have first manually remaned the column names to make them more understandable. Then, I have used the info() and describe() functions to check the details of the dataset. I have narrowed down the dataset by dropping all the irrelevant columns and rows which consisted of null value that is, '?'. Then to utilize the textual data more efficiently, I have converted the textual data into numerical data for all the relevant columns. The last and final step in preprocessing is dividing the dataset into train and test dataset which I have performed in the end.

### Three Neural Networks for experiments

We are required to implement three types of deep neural networks and with the aim to understand how performance of the neural network varies with varying combinations of hidden layers, I will be experimenting with three types of models: 

* Increasing nodes model refers to the following combination of layers: 10-> 20-> 30-> 40-> 50-> 60
* Decreasing nodes model refers to the following combination of layers: 60-> 50-> 40-> 30-> 20-> 10
* Alternating nodes model refers to the following combination of layers: 60-> 30-> 60-> 30-> 60-> 30

I am implementing a deep feed forward network with different combination of hidden layers. I tried implementing LSTM in the past but since my data is numerical I figured that Deep FeedForward network would be the best model to go for because according to what I studied online, I could understand that other neural networks like RNN, CNN, LSTM, etc are best suited for image data. Since I am working on numerical data, I believe this model would work better. Also, after experimenting with these three models, I will be experimenting with the professor's code given for our neural networks assignment to convert it into a deep neural network by playing around with the hidden layers. As we are asked to work with more than 5 layers, I am using 6 layers for the implementation. I have studied the multilayer perceptron from online sources to implement best code and I took help from the edureka website from their deep learning course, with the help of which I shall be implementing the models.

#### Selecting number of neurons in hidden layers and why I selected this model
From Introduction to Neural Networks for Java (second edition) by Jeff Heaton I could study the following concepts about selecting number of neurons in hidden layers.
Deciding the number of neurons in the hidden layers is a very important part of deciding your overall neural network architecture. Though these layers do not directly interact with the external environment, they have a tremendous influence on the final output. Both the number of hidden layers and the number of neurons in each of these hidden layers must be carefully considered.

Using too few neurons in the hidden layers will result in something called underfitting which occurs when there are too few neurons in the hidden layers to adequately detect the signals in a complicated data set. On the other hand, using too many neurons in the hidden layers can result in several problems. First, too many neurons in the hidden layers may result in overfitting. Overfitting occurs when the neural network has so much information processing capacity that the limited amount of information contained in the training set is not enough to train all of the neurons in the hidden layers. A second problem can occur even when the training data is sufficient. An inordinately large number of neurons in the hidden layers can increase the time it takes to train the network. The amount of training time can increase to the point that it is impossible to adequately train the neural network. Obviously, some compromise must be reached between too many and too few neurons in the hidden layers.

There are many rule-of-thumb methods for determining the correct number of neurons to use in the hidden layers, such as the following:

The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.

I can abide by these rules of thumb directly but in order to understand these theories I am willing to try my own selection of hidden layers to see what outcomes I can get and if these rules are really trust-worthy.

Multilayer Perceptron: A multilayer perceptron (MLP) is a class of feedforward artificial neural network. A MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

Input Nodes – The Input nodes provide information from the outside world to the network and are together referred to as the “Input Layer”. No computation is performed in any of the Input nodes – they just pass on the information to the hidden nodes.

Hidden Nodes – The Hidden nodes have no direct connection with the outside world (hence the name “hidden”). They perform computations and transfer information from the input nodes to the output nodes. A collection of hidden nodes forms a “Hidden Layer”. While a network will only have a single input layer and a single output layer, it can have zero or multiple Hidden Layers. A Multi-Layer Perceptron has one or more hidden layers.

Output Nodes – The Output nodes are collectively referred to as the “Output Layer” and are responsible for computations and transferring information from the network to the outside world.


1. I have first imported all the required libraries at the top.

2. Then I have used the dataframe where my data is stored and segregated the feature and target variables. I have also converted the categorical values into numerical values at the top. 

3. I will be using a one hot encoder function which adds extra columns based on number of labels . Thus it converts class label integers into a one-hot array where each unique label is represented as a column.

4. I have used scikit learn to split the data set into test and train sets of data.

5. I have then defined a few variables like, Learning rate which is the amount by which the weight will be adjusted. Training epochs are the number of iterations and cost history is an array that stores the cost values in successive epochs. Weight is the tensor variable to store weight values and Bias is also a tensor which stores bias values.

6. Further cost is calculated using the formula cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

7. The model is finally training based on the training_epochs and validated using the test set.

Relu activation: ReLU stands for rectified linear unit, and is a type of activation function. Mathematically, it is defined as y = max(0, x). ReLU is linear (identity) for all positive values, and zero for all negative values. It converges faster. It’s cheap to compute as there is no complicated math. The model can therefore take less time to train or run.

Sigmoid activation: The sigmoid function is used mostly used in classification type problem since we need to scale the data in some given specific range with a threshold. For instance you have two classes where you need to classify you data. using the sigmoid fuction 1/(1+ex) will adjust all your data points between 0 and 1. If you want to adjust it to 2 or a higher value, just change the numerator and you are good to go.

Linear activation: A = cx. A straight line function where activation is proportional to input ( which is the weighted sum from neuron ). It gives a range of activations, so it is not binary activation.


## Results

After I performed all the above mentioned experiments, I was able to come up with average time taken by each model, accuracy provided and also the MSE error. I have tabulated the results roughly to deduce which model works better in which circumstance.

From these results it can be said that in terms of time taken, when there were less number of nodes in hidden layers, the expanding and contracting model gave comparable results. Then on having more nodes in the hidden layers, expanding model showed lesser execution time. On having too many nodes in hidden layers, contracting model performed the best.

Now on the basis of accuracy, when there were less number of nodes in hidden layers, the expanding model gave best accuracy. Then on having more nodes in the hidden layers, alternating model showed better results. On having too many nodes in hidden layers, contracting model performed the best.

In case of MSE, when there were less number of nodes in hidden layers, the contracting model gave least error. Then on having more nodes in the hidden layers, expanding model showed better results. On having too many nodes in hidden layers, contracting model performed the best.

We can see that when we have too many nodes in hidden layers, contracting model seems to work the best out of all in all scenarios. Then expanding model and then the alternating model works better.
