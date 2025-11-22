# Import the pandas library for data manipulation and analysis
# 'pd' is a common alias/shorthand used in the data science community
import pandas as pd

# Read the CSV (Comma Separated Values) file containing salary data
# This loads the data into a DataFrame (a table-like data structure)
data = pd.read_csv("Salary_Data.csv")

# Extract the independent variable (also called feature or input variable)
# YearsExperience is what we'll use to predict salary
# Double brackets [[]] keep it as a DataFrame instead of a Series
independent = data[["YearsExperience"]]

# Extract the dependent variable (also called target or output variable)
# Salary is what we want to predict based on years of experience
# Double brackets [[]] keep it as a DataFrame instead of a Series
depedent = data[["Salary"]]

# Import the function that splits data into training and testing sets
# This is from scikit-learn (sklearn), a popular machine learning library
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
# X_train: Independent variable data for training (70% of data)
# X_test: Independent variable data for testing (30% of data)
# y_train: Dependent variable data for training (70% of data)
# y_test: Dependent variable data for testing (30% of data)
# test_size=0.30 means 30% of data goes to testing, 70% goes to training
# random_state=0 ensures we get the same random split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(independent, depedent, test_size=0.30, random_state=0)

# Import the LinearRegression class from scikit-learn
# This implements the linear regression algorithm (y = mx + b)
from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression model
# This is like creating an empty model that hasn't learned anything yet
regressor = LinearRegression()

# Train the model using the training data
# .fit() method finds the best line (calculates weight and bias) that fits the training data
# It learns the relationship between X_train (years) and y_train (salary)
regressor.fit(X_train, y_train)

# Get the coefficient (slope/weight) of the linear equation
# This tells us how much salary increases for each additional year of experience
# In the equation y = mx + b, this is the 'm' value
weight = regressor.coef_

# Get the intercept (bias) of the linear equation
# This is the base salary when years of experience is 0
# In the equation y = mx + b, this is the 'b' value
bias = regressor.intercept_

# Use the trained model to make predictions on the test data
# This predicts salaries for the X_test years of experience
# We compare these predictions with y_test to see how accurate our model is
y_pred = regressor.predict(X_test)

# Import the r2_score function to evaluate model performance
# R2 score (R-squared) measures how well our model fits the data (0 to 1, higher is better)
from sklearn.metrics import r2_score

# Calculate the R2 score by comparing actual test values (y_test) with predictions (y_pred)
# A score close to 1 means the model predicts well
# A score close to 0 means the model doesn't predict well
r_score = r2_score(y_test, y_pred)

# Import pickle library to save Python objects to files
# This allows us to save the trained model and use it later without retraining
import pickle

# Define the filename where we'll save our trained model
# .sav is just a convention; you could use .pkl or any extension
finalname = "finalized_mode_liner.sav"

# Save the trained model to a file on disk
# pickle.dump() serializes (converts) the model object into bytes
# open(finalname, 'wb') opens file in write-binary mode
# 'wb' means: w = write, b = binary mode
pickle.dump(regressor, open(finalname, 'wb'))

# Load the saved model from disk to verify it works
# pickle.load() deserializes (converts back) the bytes into a model object
# open(finalname, 'rb') opens file in read-binary mode
# 'rb' means: r = read, b = binary mode
loaded_model = pickle.load(open(finalname, 'rb'))

# Use the loaded model to predict salary for someone with 15 years of experience
# Double brackets [[15]] are used because predict() expects a 2D array
result = loaded_model.predict([[15]])

# Print the predicted salary for 15 years of experience
# This displays the result in the console
print(result)
