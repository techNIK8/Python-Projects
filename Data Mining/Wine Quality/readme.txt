Winequality-red.csv file contains wines and metrics that characterize them. 
There is also an estimate of their quality by a taster, which will be guessed using the SVM (Support Vector Machines) 
family of classification algorithms.
A. Split the dataset into training-test with a ratio of 75%-25% and measure the performance of your model using f1 score, 
precision and recall metrics. 
B. Remove 33% of the values of the ph column of the training dataset handle the missing values with the following methods:
1. Remove the column
2. Fill in the values with the average of the column items
3. Fill in the values using Logistic Regression
4. Apply K-means and fill in the missing values with the arithmetic mean of the cluster to which the sample belongs.
On the resulting new registers train an SVM with the best parameters you found in subquery A.

