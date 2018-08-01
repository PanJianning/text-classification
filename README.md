# text-classification
A text classification project using the dataset from YunYi Cup

### Problem
The data set contains 22w (x,y) records.

x is a comment text on some tourist attractions, y is the corresponding comment score range from 1 to 5

This is a supervised learning problem. At test time, given a comment, we want to predict it's score.

Since the score is an ordinal variable, it turns out that regression is more suitable than classification here.
**So the metric is mse**

Note: I use 60% of the data as training set and 40% as validation set, no test set.

### Baseline Model
Stacking: 
1. first layer:  Logistic regression and Naive Bayes
2. second layer: xgboost

validation mse: 0.4057700535879059
