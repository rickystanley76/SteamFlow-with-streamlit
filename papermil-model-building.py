import pandas as pd
import numpy as np

papermil = pd.read_csv('papermil_water_flow.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = papermil.copy()

final_df = df[['pulp_to_mixing_tank23', 'flow_after_machine_chest21', 'level_condensate_bucket1_valve_position', 'steam_group3_pressure',
              'steam_group4under_pressure', 'pressure_yankee_cylinder', 'steam_group5under_pressure',
              'steam_pressure5_over', 'pressure_condensate_bucket5', 'production_paper_machine2', 'dry_production_vira',
              'steamflow']]

X = final_df.drop('steamflow', axis=1)
y = final_df['steamflow']

# Build random forest model
# With Random Forest Regressor

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.30)

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
random_search_rf = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, cv = 10, n_jobs = -1, verbose = 2)
model_fit_RF= random_search_rf.fit(X_train, y_train)
print(model_fit_RF.best_params_)
##Testing the model
test_predict_RF = model_fit_RF.predict(X_test)


from sklearn import metrics

print('RF Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_predict_RF))
print('RF Mean Squared Error:', metrics.mean_squared_error(y_test, test_predict_RF))
print('RF R2:', metrics.r2_score(y_test, test_predict_RF))
print('RF Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_predict_RF)))
# Saving the model
import pickle
pickle.dump(model_fit_RF, open('papermil_rf.pkl', 'wb'))


#DT with hypeparmeter tuning
from sklearn.tree import DecisionTreeRegressor
## Hyper Parameter Optimization

params={
 "splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
    
}

## Hyperparameter optimization using GridSearchCV
from sklearn.model_selection import GridSearchCV
dtree = DecisionTreeRegressor()
random_search_dt=GridSearchCV(dtree,param_grid=params,scoring='r2',n_jobs=-1,cv=10,verbose=3)
model_fit_DT=random_search_dt.fit(X_train, y_train)
print(model_fit_DT.best_score_)

predictions_dt=random_search_dt.predict(X_test)

from sklearn import metrics

print('DT Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions_dt))
print('DT Mean Squared Error:', metrics.mean_squared_error(y_test, predictions_dt))
print('DT R2:', metrics.r2_score(y_test, predictions_dt))
print('DT Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions_dt)))
# Saving the model
import pickle
pickle.dump(model_fit_DT, open('papermil_dt.pkl', 'wb'))



