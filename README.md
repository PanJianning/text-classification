# text-classification
A text classification project using the dataset of YunYi Cup

### Problem
The data set contains 22w (x,y) records.

x is a comment text, y is the corresponding comment score range from 1 to 5

This is a supervised learning problem. At test time, given a comment, we want to predict it's score.

Since the score is an ordinal variable, it turns out that regression is more suitable than classification here.

**The metric is 1/(1+rmse)**

Note: We use 80% of the data as training set and 20% as validation set, no test set.

### Baseline Model
Stacking: 
1. first layer:  Logistic regression and Naive Bayes (classification)
2. second layer: xgboost (regression)

validation score(not round): 0.6124758250642371
validation score(rounded): 0.5874567866356902
