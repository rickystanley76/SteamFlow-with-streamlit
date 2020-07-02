import pandas as pd
papermil = pd.read_csv('papermil_water_flow.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = papermil.copy()
#target = 'steamflow'
#encode = ['sex','island']

#for col in encode:
 #   dummy = pd.get_dummies(df[col], prefix=col)
  #  df = pd.concat([df,dummy], axis=1)
  #  del df[col]

#target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
#def target_encode(val):
 #   return target_mapper[val]

#df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('steamflow', axis=1)
y = df['steamflow']

# Build random forest model
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()
#clf.fit(X, Y)

from sklearn.model_selection import train_test_split
#from sklearn import metrics 
#training set 80%
#testing set 20%

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.20)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV # Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 500, 1000]
}
# Create a based model
rf = RandomForestRegressor() # Instantiate the grid search model
random_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, cv = 5, n_jobs = -1, verbose = 2)
model_fit_RF= random_search.fit(X_train, y_train)
print(model_fit_RF.best_params_)
##Testing the model
test_predict_RF = model_fit_RF.predict(X_test)


# Saving the model
import pickle
pickle.dump(model_fit_RF, open('papermil_rf.pkl', 'wb'))
