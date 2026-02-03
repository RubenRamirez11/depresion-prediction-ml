# Comparison and Selection of Supervised Learning Models for the Prediction of Depression in University Students

## Description
This repository contains a supervised learning analysis aimed at
predicting the presence of depressive symptoms based on psychological
and sociodemographic variables.

Multiple classification models were evaluated using cross-validation,
with ROC-AUC as the performance metric. Based on this criterion, logistic
regression was identified as the model with the best overall performance.

## Data
The analysis uses a publicly available dataset obtained from Kaggle:
https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data

The dataset is provided in CSV format.

## Methodology
- Split of the data into training and test sets
- Cross-validation for model comparison and hiperparameter tuning
- Selection of the model with the best predictive performance
- Final training of the selected model on the full training set
- Final evaluation on the test set using ROC curves

## Model Comparison Results

The following table summarizes the performance of the evaluated models
using cross-validation. All models were tuned prior to evaluation, and
ROC-AUC was used as the performance metric.

| Model                 | Optimal Hyperparameters                          | AUC (CV) |
|-----------------------|--------------------------------------------------|----------|
| Logistic Regression   | —                                                | 0.9245   |
| RDA                   | λ = 1, γ = 0                                     | 0.9220   |
| SVM                   | C = 0.25                                         | 0.9216   |
| Gradient Boosting     | shrinkage = 0.05, depth = 1, n.trees = 200       | 0.9214   |
| AdaBoost              | mfinal = 100, maxdepth = 2                       | 0.9211   |
| Naive Bayes           | laplace = 2                                      | 0.9205   |
| Neural Network (MLP)  | size = 2, decay = 0.5                            | 0.9185   |
| Random Forest         | mtry = 2, ntree = 200, nodesize = 131            | 0.9162   |
| Bagging               | nbag = 100                                       | 0.9000   |
| KNN                   | k = 480                                          | 0.8894   |
| Decision Tree         | cp = 0.001                                       | 0.8800   |

Based on cross-validation results, logistic regression achieved the
highest ROC-AUC and was selected as the final model.

## Final Model
The final model corresponds to a binary logistic regression,
trained using all observations from the training set and subsequently
evaluated on the test set to estimate its generalization performance.

## Final Test Performance

The selected logistic regression model was retrained using the full
training dataset and evaluated on an independent test set.  
The final model achieved a ROC-AUC of **0.9173** on the test data,
indicating good generalization performance.

## Reproducibility
To reproduce the analysis, run the `modelo.R` script from the root
directory of the repository.

