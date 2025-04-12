import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"Random Forest with Rookie Scale Contract Feature")
print(f"")

# loading data 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full_data = pd.read_csv(os.path.join(project_root, 'data', 'full_data.csv'))

def rookie_scale(age):
    experience = age - 20
    if 1 <= experience <= 4:
        return 1
    else:
        return 0 

full_data['rookie_scale'] = full_data['Age'].apply(rookie_scale)
test_season = 'season_2023-24'
train_data = full_data.loc[full_data[test_season] == 0].copy()
test_data = full_data.loc[full_data[test_season] == 1].copy()
X = train_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
Y = train_data['Cap_Pct']
new_X = test_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
new_Y = test_data['Cap_Pct']

# model evaluation 
param_grid = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [7, 8, 9],
    'max_features': [0.33, 0.35]
}

rf = RandomForestRegressor(random_state=32)
tscv = TimeSeriesSplit(n_splits=3)
grid_seach = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_seach.fit(X, Y)
print(f"Tunning Hyperparameters via Grid Search CV")
print(f"Best Parameters: {grid_seach.best_params_}")
print(f"Best Score: {grid_seach.best_score_:.5f}")
print(f"")

# fitting model on full data and predicting on unseen data
random_forest_mod = grid_seach.best_estimator_
new_pred = random_forest_mod.predict(new_X)
print(f"Scores for Prediction on 2023-24 Data:")
print(f"MAE: {mean_absolute_error(new_Y, new_pred):.5f}") # 0.02710
print(f"MSE: {mean_squared_error(new_Y, new_pred):.5f}") # 0.00162
print(f"")

model_filename = 'random_forest_model_with_rookie.joblib'
# joblib.dump(random_forest_mod, os.path.join(project_root, 'models', model_filename))

# feature importance report
feature_names = X.columns
feature_importance_score = random_forest_mod.feature_importances_
score_dict = dict(zip(feature_names, feature_importance_score))
feature_importance_score_df = pd.DataFrame({"feature": feature_names, "score": feature_importance_score})
feature_importance_score_df.sort_values(by='score', ascending=False, inplace=True)
print(f"Top 5 Features By Importance:")
print(f"{feature_importance_score_df.head()}")
print(f"")

# final predictions on unseen data
new_pred_series = pd.Series(new_pred)
test_data_copy = test_data.copy()
test_data_copy.reset_index(inplace=True)
test_data_copy['Predicted_Cap_Pct'] = np.round(new_pred_series, 4)
test_data_copy['Predicted_Salary'] = np.round(test_data_copy['Predicted_Cap_Pct']*test_data_copy['Salary_cap'], 0)
final_preds = test_data_copy[['player', 'Cap_Pct', 'Predicted_Cap_Pct', 'Salary', 'Predicted_Salary']].copy()
final_preds[['Salary', 'Predicted_Salary']] = final_preds[['Salary', 'Predicted_Salary']].map(lambda x: f"${x:,.0f}")
print(f"Salary Predictions:")
print(f"{final_preds.sample(5)}")

