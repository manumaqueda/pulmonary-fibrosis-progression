Capstone project - Machine Learning Nanodegree
===========

Manuel Maqueda Vinas, 15 August 2020

## Definition

### Project Overview
Pulmonary fibrosis, a disorder with no known cause and no known cure, created by scarring of the lung. Outcomes of the 
disease can vary from long-term stability to rapid deterioration, but doctors are not easily able to understand the 
exact outcome in advance.

In addition some of the current methods implies long times and substantial effort in order to produce an accurate 
prognose. This increases patients anxiety and delays the application of the right mitigation action.  

In this project I am creating a model which can help doctors and patients to understand in advance the evolution of 
the disease by predicting lung capacity of the patients in future weeks.

This project is based in the *OSIC pulmonary fibrosis progression* [Kaggle competition](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)


### Problem Statement 
The goal of the project is to predict patients' lung capacity based on a given train dataset. The main target is to predict
weekly FVC measurements for 5 patients in the following 2-3 years.

In order to achieve the previous goal, this project is divided in the following sections:

* Exploration Data Analysis 
    * Understanding both data flavours, tabular and images, in the given dataset
    * Exploring tabular data in details to have a good intuition of the information provided
* Data preparation
    * Extracting main features for the modelling part
    * Normalization of data features
    * Preparing the data for the training and test exercise
* Modelling and Training
    * Defining a custom neural network implemented in PyTorch
    * Creating a training algorithm using a quantile regression strategy
    * Showcasing the competition evaluation metric with a validation dataset
* Model deployment and inference
    * Upload the code in AWS SageMaker for training and model endpoint deployment
    * Performing inference using the test data in the competition
    * Understanding predictions 
    
It is also worth mentioning that some other improvements are out of the scope of this project due to timing 
constraints. Those are described in the Future Work section at the end of this report.

### Metrics

In order to create our model a couple of metrics has been used, `quantile loss` for the backpropagation step
when training the PyTorch neural network and the other a modified version of the `Laplace Log Likelihood (LLL)` which
is used by the Kaggle competition to evaluated the submissions. In our case the `LLL` has been implemented just 
on the validation dataset in other to illustrate its implementation and to enable further future work to control
training epochs based on that metric.

#### Quantile Loss
The implementation in this project is based in the definition of the quantile loss, available [here](https://www.wikiwand.com/en/Quantile_regression)

#### Laplace Log Likehood
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20%5Csigma_%7Bclipped%7D%20%3D%20%5Cmax%20%5Cleft%20(%20%5Csigma%2C%2070%20%5Cright%20)%20%5C%5C"><br/><br/>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20%5CDelta%20%3D%20%5Cmin%20%5Cleft%20(%20%5C%7CFVC_%7Bture%7D%20-%20FVC_%7Bpredicted%7D%5C%7C%2C%201000%20%5Cright%20)%20%5C%5C%0A"><br/><br/>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20f_%7Bmetric%7D%20%3D%20-%20%5Cfrac%7B%5Csqrt%7B2%7D%20%5CDelta%7D%7B%5Csigma_%7Bclipped%7D%7D%20-%20%5Cln%20%5Cleft(%20%5Csqrt%7B2%7D%20%5Csigma_%7Bclipped%7D%20%5Cright)%20."/><br/><br/>


## Analysis

### Data Exploration
    
### Exploratory Visualization
    
### Algorithms and Techniques
    
### Benchmark

## Methodology
https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
https://www.kaggle.com/carlossouza/quantile-regression-pytorch-tabular-data-only

### Data Preprocessing
    
### Implementation

### Refinement

## Results

### Model Evaluation and Validation
    
### Justification

## Future Work

## References

    
