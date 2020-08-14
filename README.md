# Pulmonary Fibrosis Progression

@Author: Manuel Maqueda Vinas

@Date: 14 August 2020


This repository has been setup in order to complete the capstone project of the Udacity Machine Learning Nanodegree
program. 

The capstone project has been selected from this [Kaggle competitiion](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression). 
In terms of this first iteration I have focused the effort into analysing the tabular data and setup a customer PyTorch
neural network to illustrate how to implement a NN in pytorch and deploy it in a cloud environment. 

Further work is required in order to include new features from the CT scan images provided as part of the input data in 
the competition and also to do some code refactoring to make it a bit more modular and to allow both, to develop and
 debug algorithm in local and in the cloud.
  
### How to set up the repository

The following steps are required in order to setup the project in a SageMaker JupyterNotebook instance:

* Clone repo -> `git clone https://github.com/manumaqueda/pulmonary-fibrosis-progression.git`
* Open a console for both kernels: `EDA-feature-extraction` and `Training-and-inference-pytorch`  Run the following command ->
`git install -r requirements.txt`. This will ensure all the libraries are setup in the virtual environment in which the 
notebooks are running
* Create a directory under the project called `data` and upload the following files from the
 [competition input data](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data). Make sure the following files are 
 updated in the directory: 
  * `train.csv`
  * `test.csv`
  * `sample_submission.csv`

### Execute Notebooks

This repository contains two notebooks that need to be executed in order. I also recommend to execute them in a classic 
Jupyter Notebook instead that on the lab version, as the majority of the graphs won't work on the lab version.

* `EDA-feature-extraction`
* `Training-and-inference-pytorch`

### Libraries Used
Below I reference the main libraries used in this project:
* [Pandas](https://pandas.pydata.org/)
* [Plotly](https://plotly.com/)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)
* [Seaborn](http://seaborn.pydata.org/)

### Kaggle Notebooks
Some of the concepts and techniques used in this project has been inspired from some existing notebooks from Kaggle. I 
need to thank all the contributors for the work done beforehand:

* https://www.kaggle.com/piantic/osic-pulmonary-fibrosis-progression-basic-eda
* https://www.kaggle.com/havinath/eda-observations-visualizations-pytorch
* https://www.kaggle.com/andradaolteanu/pulmonary-fibrosis-competition-eda-dicom-prep
* https://www.kaggle.com/ulrich07/osic-multiple-quantile-regression-starter
* https://www.kaggle.com/twinkle0705/your-starter-notebook-for-osic
* https://www.kaggle.com/avirdee/understanding-dicoms
