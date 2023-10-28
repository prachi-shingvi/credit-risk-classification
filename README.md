# Challenge 20 : Credit Risk Classification using Logistic Regression

## Overview of the Analysis

- In the given challenge, we were provided a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. We had to then split the data into 2 groups viz. training set and testing set to train the model and predict the testing set respectively.
- The data contained important information regarding the borrower's loan size, interest rate, income, debt to income ratio, number of accounts, derogatory marks and current total debt of the borrower. With these parameters, we were given loan status denoted by 0(indicating healthy loan) and 1 (indicating high risk of defaulting) which was the binary prediction we had to make.
- We used Pandas dataframes to read the csv and found that there were 75036 healthy loan samples and 2500 high risk ones using `value_counts`.
- The following steps were taken to train and test the first model:
  1. We split the data using `train_test_split` method inside `sklearn.model_selection`. The `train_test_split` method returned training data(X_train, y_train) and testing data(X_test, y_test).
  2. We created a logistic regression model using `sklearn.linear_model` python library's `LogisticRegression` method and used training data to fit the model.
  3. We then used testing data(X_test) to predict the results(testing for loan status).
  4. We evaluated the model's performance using balanced accuracy score, confusion matrix and classification report.
- After this, we created another model to resolve imbalances in data by resampling using `RandomOverSampler`. This helped us receive 56271 samples to train our model. This method helped boost the accuracy of the model.


## Results
- Machine Learning Model 1(without using RandomOverSampler):
1. Accuracy: We had **95.20%** accuracy
2. Precision: For healthy loan(0), we had a precision of 1.00, however, for default risk loans(1), we had precision of 0.85
3. Recall: For healthy loan(0), we had a recall of 0.99, however, for default risk loans(1), we had recall of 0.91


* Machine Learning Model 2(using RandomOverSampler):
1. Accuracy: We had **99.36%** accuracy
2. Precision: For healthy loan(0), we had a precision of 1.00, however, for default risk loans(1), we had precision of 0.84
3. Recall: For healthy loan(0), we had a recall of 0.99. Similarly, for default risk loans(1), we had recall of 0.99

## Summary

- Looking at the results, we can definately say that accuracy of the model increased after resampling the data from 95.20% to 99.36%. (Model 2 performed better than Model 1)
- Looking at classification report, both models predicted healthy loans with really high accuracy. However, they both had a lower accuracy percentage while predicting default risk loans.
- Additionally, it is important to identify 1s as 1s(default risking) than having a few 0s(healthy loans) being predicted as 1s(false positives). Hence, our model 2 will be considered better as it correctly predicted 615/619 samples as compared to 563/619 for model 1.
