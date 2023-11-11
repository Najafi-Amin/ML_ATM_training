#@author: Amin
#Description: The following program reads a csv file
#and trains ML models used for future prediction

# Importing necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Read file
dataset = pd.read_csv('PS_20174392719_1491204439457_log_reduced.csv')
dataset.head()

# Store numeric columns
numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']



# Boxplot
fig, ax = plt.subplots(figsize=(15, 6))
plt.boxplot(x=dataset[numeric_columns])
ax.set_xticklabels(numeric_columns, rotation=30)
ax.set_yticks(range(0, 50000000, 10000000))
ax.set_yticklabels(['$0', '$10M', '$20M', '$30M', '$40M'])
plt.title('Data', fontsize=15)
plt.show()


from sklearn.preprocessing import Normalizer
normalizer = Normalizer() #instantiate object

# Creating a deep copy of our dataframe so it may be modified without affecting the original
# For convenience, only the columns to be used in training our model will be kept
columns_to_keep = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
dataset_copy = dataset[columns_to_keep].copy(deep=True)

# Replacing the numeric columns with their normalized values
dataset_copy[numeric_columns] = normalizer.fit_transform(dataset_copy[numeric_columns])

# Boxplot of Transformed Numeric Variables
plt.clf()
fig, ax = plt.subplots(figsize=(10, 5))
plt.boxplot(x=dataset_copy[numeric_columns])
ax.set_xticklabels(numeric_columns, rotation=30)
plt.ylabel('z-Score')
plt.title('Boxplot of Dataset\'s Transformed Numeric Variables', fontsize=15)
plt.style.use('ggplot')
plt.show()

# One-hot encoding 'type' column
dataset_copy = pd.get_dummies(dataset_copy, columns=['type'], prefix=['type'])

# Importing the library to split our data
from sklearn.model_selection import train_test_split

# Separating the dataset
dataset_copy_y = dataset_copy['isFraud']
dataset_copy_X = dataset_copy.drop(labels='isFraud', axis=1)

#Assign data to train and test groups
X_train, X_test, y_train, y_test = train_test_split(dataset_copy_X, dataset_copy_y)

# Importing the Logistic Regression machine learning library
from sklearn.linear_model import LogisticRegression

# Creating a Logistic Regression instance
logistic_regression_model = LogisticRegression(max_iter=500)

# Training the Logistic Regression model on our test data
logistic_regression_model.fit(X_train, y_train)

# Importing the Decision Tree machine learning library
from sklearn.tree import DecisionTreeClassifier

# Creating a Decision Tree instance
decision_tree_model = LogisticRegression(max_iter=500)

# Training the Decision Tree model on our test data
decision_tree_model.fit(X_train, y_train)

# Importing the Random Forest machine learning library
from sklearn.ensemble import RandomForestClassifier

# Creating a Random Forest instance
random_forest_model = RandomForestClassifier()

# Training the Random Forest model on our test data
random_forest_model.fit(X_train, y_train.values.ravel())

# Displaying fraud percentage from data
print(f'The fraud percentage from data {dataset["isFraud"].sum() / len(dataset)}')

# Importing the recall calculation library
from sklearn.metrics import recall_score

# Making predictions for our test data using our trained models
lr_y_pred = logistic_regression_model.predict(X_test)
dt_y_pred = decision_tree_model.predict(X_test)
rf_y_pred = random_forest_model.predict(X_test)

# Printing our results
print('Logistic Regression Recall:', recall_score(y_test, lr_y_pred))
print('Decision Tree Recall:', recall_score(y_test, dt_y_pred))
print('Random Forest Recall:', recall_score(y_test, rf_y_pred))
