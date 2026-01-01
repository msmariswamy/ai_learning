import pandas as pd

dataset = pd.read_csv('../4. Decision tree/50_Startups.csv')

dataset = pd.get_dummies(dataset, drop_first=True)

independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend',
       'State_Florida', 'State_New York']]

dependent = dataset[['Profit']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

from sklearn.svm import SVR
#regressor = SVR(kernel='linear', C=0.01, gamma=0.1, epsilon=0.1)
regressor = SVR(kernel='rbf', C=0.01, gamma=0.1, epsilon=0.1)

regressor.fit(X_train, y_train)


#weight = regressor.coef_
origin = regressor.n_support_
support = regressor.support_
print(support)
bias = regressor.intercept_


y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 Score of Multiple Linear Regression Model is:", r2)
#R2 Score of Multiple Linear Regression Model is: 0.9358680970046243

import pickle
fileName = '../4. Decision tree/finalized_model_multiple.sav'
pickle.dump(regressor, open(fileName, 'wb'))

loaded_model = pickle.load(open(fileName, 'rb'))
results = loaded_model.predict([[1234,345,4565,1,0]])
print(results)

# this gave us only -0.05, which is bad, so lets do standardazation in next part