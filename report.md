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
This project implementation relies in the quantile loss metric to train the custom neural network. There is more documentation
 available about this metric [here](https://www.wikiwand.com/en/Quantile_regression)
 
 *TODO: formula quantile loss*

#### Laplace Log Likehood
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20%5Csigma_%7Bclipped%7D%20%3D%20%5Cmax%20%5Cleft%20(%20%5Csigma%2C%2070%20%5Cright%20)%20%5C%5C"><br/><br/>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20%5CDelta%20%3D%20%5Cmin%20%5Cleft%20(%20%5C%7CFVC_%7Bture%7D%20-%20FVC_%7Bpredicted%7D%5C%7C%2C%201000%20%5Cright%20)%20%5C%5C%0A"><br/><br/>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20f_%7Bmetric%7D%20%3D%20-%20%5Cfrac%7B%5Csqrt%7B2%7D%20%5CDelta%7D%7B%5Csigma_%7Bclipped%7D%7D%20-%20%5Cln%20%5Cleft(%20%5Csqrt%7B2%7D%20%5Csigma_%7Bclipped%7D%20%5Cright)%20."/><br/><br/>


## Analysis
### Data Exploration
In this part of the project the provided data has been analysed, focusing in the tabular data as mentioned above. 

The first analysis performed is in the `train.csv` dataset. Below it is shown some of the characteristics derived from the analysis:
* Number of rows is 1549
* Number of unique patients is 176, which means that each patient has several rows corresponding to each of the individual 
week visits to measure the FVC.
* Smoking values are `Ex-smoker` `Never smoked` and `Currently smokes` 
* Each of the rows contain the following data:
 * Weeks- the relative number of weeks pre/post the baseline CT (may be negative)
 * FVC - the recorded lung capacity in ml
 * Percent- a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
 * Age
 * Sex (Gender)
 * SmokingStatus
 
*TODO- INSERT capture of train_df.head()* 

*TODO- INSERT capture of patient_df.head()* 

In terms of the imaging data the following has been verified in order to support any future work:
* Every patient in the tabular data has a directory containing the images of the baseline CT scan.

### Data Visualization
In this section an inspection of the tabular data has been performed in order to obtain more details of the data distribution
per each of the available features: FVC, percent, age, sex, smoking status and weeks.

#### Age and Smoking distribution over Sex
As it is shown in the figure from below, the majority of the patients in the training dataset are males, and also they 
are aged between 60 and 75 years old. In addition we can see that more than the 50% of the patients are also `Ex-smokers`

*TODO - insert age distribution over sex*

*TODO - insert smoking distribution over sex*

#### FVC (lung capacity) per Age, Sex and Smoking Status
When looking at the FVC distribution over Age, Sex and Smoking Status, it is observed that in overall older people tend
 to have less lung capacity (FVC) and that average lung capacity is substantially lower for females than for males. These
results are not a surprise and it might be more linked to the biological capacity of females and aged people rather than
the actual incidence of the disease and the potential evolution of itself.

Another conclusion is that it is not obvious that ex-smokers or current smokers have less lung capacity than people that 
never smoked. In order to understand that in more detail this data has been segmented in the last part of this section
 to analyse Percent by Sex.

*TODO - insert FVC per age*
*TODO - insert FVC per sex*
*TODO - insert FVC per smoking status*

#### Percent (lung capacity) per Age, Sex and Smoking Status
Looking at the distribution of percent against the rest of the properties seem to be the most relevant to actually understand
the distribution of the lung capacity of the patients in the train dataset.

As part of the conclusions looking at percent distributions, it is observed that in overall younger patients have less
lung capacity of what it is expected for them. Also the capacity is balance per sex, which indicates that there is no evidence
that a particular sex is more prone than the other to have a more rapid deterioration. In addition, it is observed again that
there is not evidence that `ex-smoker` or `smoker` persons have less lung capacity than `never smoked` persons.

*TODO - insert percent per age*
*TODO - insert percent per sex*
*TODO - insert FVC percent smoking status*

#### Percent (lung capacity) and Smoking Status per both sex: female and male
In the following two graphs it is shown the distribution of percent per smoking status and particular gender. Some of the
observations that are derived from both images are that we can confirm there is no evidence that smokers or ex-smokers have
less lung capacity than never smoked person. However, it is worth mention that the evolution of the capacity might not follow
the same trend. That is also mentioned as part of the future work.

*TODO - insert male percentage over smoking status*
*TODO - insert female percentage over smoking status*

#### Feature Correlation
As part of the analysis of the tabular data, a correlation matrix has been generated. From that matrix we can observer 
the low correlation between our features apart from FVC and Percentage. This correlation between percentage and FVC is
 expected as both are measuring lung capacity. It is also important to note that there is a bit of correlation between 
 SmokingStatus and Percent which might indicate that *it might be* a relationship between both features over time, implying 
 that there might be an impact on the SmokingStatus in the evolution of the disease.

*TODO - insert correlation image*

### Algorithms and Techniques
The first thing that done is to select the features to be used and preprocess it properly, including the normalization of
the input variables and preparing the dataset to be consumed by the training algorithm. 

In order to obtain lung capacity prediction a custom neural network has been created using PyTorch. This is a 3-layered 
network which uses a quantile regression technique in order to adjust its weights as part of the training exercise.

*TODO - create image of 3-layered nn*

In particular we have used the two metrics explained above, the first one quantile error is used for the backpropagation
in of the neural network and the second one Laplace Log Likehood it has been just added to illustrate with a validation 
dataset how it is improved over the training exercise since this is the metric used by the Kaggle competition to evaluate
the solution.

As part of the inference exercise the trained model provides three different values per each of the quantiles specified in 
training exercise. This is used to obtain the main prediction but also the confidence of the prediction which is just calculated
as the difference between the inference of the higher quantile minus the inference corresponding to the lower quantile.

The model trained has been also tested as a model endpoint in the cloud using AWS Sagemaker to deploy it.

### Benchmark

## Methodology

### Data Preprocessing
    
### Implementation

### Refinement

## Results

### Model Evaluation and Validation
    
### Justification

## Future Work
* Evolution of capacity over smoking status

## References
https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
https://www.kaggle.com/carlossouza/quantile-regression-pytorch-tabular-data-only

    
