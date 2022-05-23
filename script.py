import pandas as pd
import numpy as np
from numpy.lib.npyio import savetxt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# List to store the labels(Normal/Abnormal or 0/1)
labels = []
# List to store the pricing curves of the training dataset
p_curves = []
# List to store the pricing curves of the testing dataset
test_data = []
# Dict to store the produced schedule
schedule = {}
# Dict to store the pricing curve value to the corresponfing hour
cost = {}
hours = []
list = [0]*24

# Assigning the datasets 
training_set = "assets/TrainingData.txt"
testing_set = "assets/TestingData.txt"
task_sheet = "assets/COMP3217CW2Input.xlsx"

# Going through every line/entry in the training datasets
for line in open(training_set):

    # Putting each line(price curves + label) in an array
    array_line = np.asarray(line.split(','))

    # Adding the labels to the label list
    labels.append(array_line[-1].astype(int))

    # Adding the pricing curves to a list
    p_curves.append(array_line[:-1].astype(float))

# Going through every line/entry in the testing datasets and putting each line in an array
test_data = [np.asarray(line.split(",")).astype(float) for line in open(testing_set)]


# Parse user & tasks data / used for Scheduling later
users_tasks = pd.read_excel(task_sheet, sheet_name="User & Task ID")


# Splitting the training dataset into random 70% train and 30% test subsets
xTrain, xTest, yTrain, yTest = train_test_split(p_curves, labels, test_size=0.3, random_state=15)

# Initialize a SVC model
model = SVC()
model.fit(xTrain, yTrain)

# Predict the test data
test_pred = model.predict(xTest)

# Predict the training data
train_pred = model.predict(xTrain)

# Printing the accuracy for the training and testing sets after predicting with SVM
print("Training prediction accuracy: ","{0:.1%}".format(
                  accuracy_score(train_pred, yTrain)))
print("Test prediction accuracy: ","{0:.1%}".format(
                  accuracy_score(test_pred, yTest)))


# Predicting outcome based on SVM model
test_label = model.predict(test_data).reshape(100, 1)
# Predicted number of abnormal prices
abnormals = np.count_nonzero(test_label == 1)
print("Predicted number of abnormal price curves: ", abnormals)


# Feed prediction to Input data
output = np.append(
test_data, test_label, axis=1)

"""
# Save the testing results
np.savetxt("TestingResults.txt", output, delimiter=",",
                   fmt=','.join(['%.14f']*24 + ['%i']))
"""

""" This function plots the results of the SVM model
            and saves it as a PNG.
"""




# Assign pricing curve values to corresponding hour
for hour in range(24):
    cost[hour] = test_data[hour]
    schedule[hour] = []
    hours.append(hour)


#print(cost)
#print("This is schedule: ", schedule)
#print("This is hours: ", hours)

# Dict comprehensive for storing the sorted data
prices = {hour: price for hour, price in sorted(
        cost.items(), key=lambda item: item[1])}


# Iterate through the users and assign tasks to hours with lowest cost
for user in users_tasks:
    for task in user.iterrows():
        demand = task["Energy Demand"]
        for hour in prices.items():
            # Check if a task has been completed
            if demand <= 0:
                break
            if task["Ready Time"] <= hour <= task["Deadline"]:
                list[hour] += min(demand,task["Maximum scheduled energy per hour"])
                demand -= task["Maximum scheduled energy per hour"]
                schedule[hour].append(task["User & Task ID"])







