# Import pickle library to load saved Python objects from files
# Pickle allows us to deserialize (load) previously saved models
import pickle

# Load the trained linear regression model from the saved file
# pickle.load() reads the binary file and converts it back into a Python object
# open("finalized_mode_liner.sav", 'rb') opens the file in read-binary mode
# 'rb' means: r = read, b = binary (because the model was saved in binary format)
# This loads the model that was trained and saved earlier
load_model = pickle.load(open("finalized_mode_liner.sav", 'rb'))

# Use the loaded model to predict salary for someone with 25 years of experience
# Double brackets [[25]] create a 2D array, which is required by the predict() method
# The model will use the equation it learned (y = mx + b) to calculate the salary
result = load_model.predict([[25]])

# Print the predicted salary for 25 years of experience
# This displays the result (predicted salary) in the console
print(result)
