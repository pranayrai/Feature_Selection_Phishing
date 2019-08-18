import pandas as pd
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


#Read the data file

with open('complteData.csv') as csvFile:
    csv_reader = csv.reader(csvFile)
    csv_headings = next(csv_reader)

# Create a dataframe with the dataset

df1 = pd.read_csv('complteData.csv', delimiter=',')

# Create a variable for Input Data - X and Result - Y

X = df1.iloc[:, 0:30] # First 30 columns
Y = df1.iloc[:, -1] # Last Column

# Code snippet to create Feature Importance Graph
# This code generates the graph

model = ExtraTreesClassifier()
model.fit(X,Y)
scores = model.feature_importances_
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()

#Select the top 6 important features
important_features = ['SSLfinal_State','URL_of_Anchor', 'having_Sub_Domain', 'Prefix_Suffix', 'web_traffic', 'Links_in_tags']

#X contains the input dataset and Y contains the result column
X= df1[important_features]
Y= df1['Result ']

#Split the dataset into 80 - training - 20 - testing
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.20)

#Create the model
model = DecisionTreeClassifier()
model = model.fit(X_train, Y_train)


prediction = model.predict(X_test)
print('Accuracy of Decision Tree Classifier:' + str(accuracy_score(Y_test, prediction)*100))

#Perform 10-fold cross value and get accuracy
CF = cross_val_score(model, df1, Y, scoring='f1_macro', cv=10)
cf_accuracy = CF.mean()
print('New Accuracy with 10-fold cross:' + str(cf_accuracy*100))

#Code to get confusion matrix
actual = Y_test
predicted = prediction
results = confusion_matrix(actual, predicted)
print( 'Confusion Matrix :')
print(results)
print ('Report : ')
print (classification_report(actual, predicted))