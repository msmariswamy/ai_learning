# Import numpy library for numerical operations and array manipulations
# 'np' is the standard alias for numpy
import numpy as np

# Import matplotlib's pyplot module for creating visualizations and plots
# 'plt' is the standard alias for pyplot
import matplotlib.pyplot as plt

# Import pandas library for data manipulation and analysis
# 'pd' is the standard alias for pandas
import pandas as pd

# Read the CSV (Comma Separated Values) file containing salary data
# This loads the data into a pandas DataFrame (a table-like structure)
# The data contains years of experience and corresponding salaries
data = pd.read_csv('Salary_Data.csv')

# Print the entire dataset to see what data we're working with
# This helps us understand the structure and content of our data
print(data)

# Extract the independent variable (feature/input) from the dataset
# YearsExperience is what we'll use to predict salary
# Double brackets [[]] keep it as a DataFrame (not a Series)
# Data type is float (decimal numbers)
independent = data[["YearsExperience"]]

# Extract the dependent variable (target/output) from the dataset
# Salary is what we want to predict based on years of experience
# Double brackets [[]] keep it as a DataFrame (not a Series)
# Data type is float (decimal numbers)
dependent = data[["Salary"]]

# Create a scatter plot to visualize the relationship between experience and salary
# Each dot represents one person's experience and salary
# This helps us see if there's a linear relationship
plt.scatter(independent, dependent)

# Set the label for the x-axis (horizontal axis) with font size 20
# This makes it clear what the horizontal axis represents
plt.xlabel('YearsExperience', fontsize=20)

# Set the label for the y-axis (vertical axis) with font size 20
# This makes it clear what the vertical axis represents
plt.ylabel('Salary', fontsize=20)

# Display the plot in a window
# This opens a new window showing the scatter plot
plt.show()

# Import the function that splits data into training and testing sets
# This comes from scikit-learn, a machine learning library
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
# X_train: Independent variable for training (66.67% of data)
# X_test: Independent variable for testing (33.33% of data)
# y_train: Dependent variable for training (66.67% of data)
# y_test: Dependent variable for testing (33.33% of data)
# test_size=1/3 means approximately 33% goes to testing, 67% to training
# random_state=0 ensures reproducible results (same split every time)
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=1/3, random_state=0)

# Print the test set's actual salary values
# This shows what salaries the model should predict for the test data
print(y_test)

# Import the LinearRegression class from scikit-learn
# This class implements the linear regression algorithm (finds best-fit line)
from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression model
# This creates an empty model that hasn't been trained yet
regressor = LinearRegression()

# Train the model using the training data
# The fit() method calculates the best weight (W) and bias (b0) values
# For the equation: y = W * x + b0
# W (weight/slope) shows how much salary increases per year of experience
# b0 (bias/intercept) is the base salary when experience is 0
regressor.fit(X_train, y_train)

# Get the coefficient (weight/slope) from the trained model
# This is the 'W' in the equation y = W * x + b0
# It represents how much salary changes for each year of experience
weight = regressor.coef_

# Print the weight value with descriptive text
# .format() inserts the weight value into the string
print("Weight of the model={}".format(weight))

# Get the intercept (bias) from the trained model
# This is the 'b0' in the equation y = W * x + b0
# It represents the predicted salary at 0 years of experience
bais = regressor.intercept_

# Print the intercept value with descriptive text
# .format() inserts the bias value into the string
print("Intercept of the model={}".format(bais))

# Use the trained model to predict salaries for the test data
# The model applies the learned equation to X_test values
# These predictions will be compared with actual y_test values
y_pred = regressor.predict(X_test)

# Print all the predicted salary values for the test set
# This shows what the model thinks the salaries should be
print(y_pred)

# Import the r2_score function to measure model accuracy
# R2 score ranges from 0 to 1, where 1 is perfect prediction
from sklearn.metrics import r2_score

# Calculate the R2 score by comparing actual vs predicted values
# y_test contains the actual salaries
# y_pred contains the model's predicted salaries
# Higher R2 score means better model performance
r_score = r2_score(y_test, y_pred)

# Print the R2 score to see how well the model performs
# Values close to 1 indicate good predictions
print(r_score)

# Import pickle library to save/load Python objects
# This allows us to save the trained model for later use
import pickle

# Define the filename for saving the model
# 'finalized_model.sav' is where the trained model will be stored
filename = 'finalized_model.sav'

# Save the trained model to a file on disk
# pickle.dump() converts the model object into bytes and saves it
# 'wb' means write in binary mode
pickle.dump(regressor, open(filename, 'wb'))

# Load the saved model from the file to verify it works
# pickle.load() reads the file and converts bytes back to a model object
# 'rb' means read in binary mode
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

# Test the loaded model by predicting salary for 15 years of experience
# [[15]] creates a 2D array as required by predict()
result = loaded_model.predict([[15]])

# Print the predicted salary for 15 years of experience
print(result)

# Get user input for making a custom prediction
# int() converts the text input into an integer number
# The user will type a number representing years of experience
prediction_input = int(input("Enter the Prediction input value:"))

# Use the model to predict salary based on user's input
# [[prediction_input]] wraps the input in a 2D array format
# The model applies its learned equation to make the prediction
Future_Prediction = regressor.predict([[prediction_input]])

# Print the predicted salary for the user's input
# .format() inserts the prediction into the string
print("Future_Prediction={}".format(Future_Prediction))
