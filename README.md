# CIBMTR_post_hct_survival

Collaboration to improve prediction of transplant survival rates equitably for allogeneic HCT patients. Inspired by the Kaggle competition [Equity in Post-HCT Survival Predictions](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/overview).

## Team Members

[Ruibo Zhang](https://www.linkedin.com/in/ruibo-zhang-b901161a1/), [Chi-Hao Wu](https://www.linkedin.com/in/chi-hao-wu-69a590227), [Yang Li](https://www.linkedin.com/in/yang-li-9018571b9/), [Ray Karpman](https://www.linkedin.com/in/rachel-karpman/), [Elzbieta Polak](https://www.linkedin.com/in/elzbieta-polak)

## Executive Summary

Please see [this google doc](https://docs.google.com/document/d/1mQpeUD_LRBRP4hCrkkFXRa48OGYRKUpNPlSAw4oAM6Q/edit?usp=sharing) for a brief summary of our project.

## Project Checkpoints
### Checkpoint 1: Dataset description

Please refer to this [google doc](https://docs.google.com/document/d/1kk4Rym6FYjPDXt6weWp9CQHLJprUmqq-LM3-46bOarc/edit?usp=sharing) or this [markdown file](Checkpoint1DatasetDescription.md).

### Checkpoint 2: EDA and data preprocessing

Please refer to [this document](Checkpoint2EDADataCleaning.ipynb). The [scripts](/scripts) directory contains notebooks with our full [univariate analysis](scripts/univariate_distributions_revised.ipynb) and [pairwise analysis](scripts/correlations_for_checkpoint.ipynb). 

### Checkpoint 3: Modeling approach

Please see [this document](Checkpoint3ModelingApproach.md) for an overview of our modeling approach. 

## Repository Organization

### Root Directory

The root directory of this repo contains four notebooks, containing our code for model selection and evaluation.
* [model_comparison.ipynb](model_comparison.ipynb) contains our model-selection process. Each potential model is evaluated using five-fold cross-validation on the [training set](data/train_set.csv). 
* [implementation.ipynb](implementation.ipynb) contains helper functions for creating and fitting pipelines and models.
* [model_comparison_test_naive.ipynb](model_comparison_test_naive.ipynb) is a script for testing each model we considered on the [test data](data/test_validation_set.csv), using a simple preprocessing pipeline which imputes -1 for missing numerical values and "missing" string for missing categorical values.
* [model_comparison_test_tuned.ipynb](model_comparison_test_tuned.ipynb) is a script for testing each model we considered on the test data, using a more complex preprocessing pipeline which uses KNN imputation and scales numerical variables.

Note: we selected a modeling approach based on cross-validation on the training set, before evaluating any model on the test set. The final metrics reported for this project are based on this chosen approach--not on the approach that performed best on test data!

### Data Directory

The [data](/data) directory contains raw data for this project. 
* [train.csv](/data/train.csv) is the full training dataset provided on Kaggle.com.
* [train_set.csv](/data/train_set.csv) is the subset of [train.csv](/data/train.csv) we used as training data.
* [test_validation_set.csv](/data/test_validation_set.csv) is the held-out data we used for final evaluation.

We assigned 80% of the data for training, and 20% for testing.

### Examples Directory

The [examples](/examples) directory contains scripts for various utility functions, including the following.

* [data_train_test_split.ipynb](examples/data_train_test_split.ipynb) was used to split our data into train and test sets.
* [concordance_index.ipynb](examples/concordance_index.ipynb) is a helper script provided on Kaggle which evaluates predictions using the stratified concordance index, a specialized evaluation metric.

### Scripts Directory

The [scripts](/scripts) directory contains additional notebooks with our exploratory analysis, and experiments with individual models and proprocessing techniques on the training data. 


