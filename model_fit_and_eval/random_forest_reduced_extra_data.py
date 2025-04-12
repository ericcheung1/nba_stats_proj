import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"Random Forest with Variable Selection from unscaled Lasso Model")
print(f"")

# loading data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full_data = pd.read_csv(os.path.join(project_root, 'data', 'full_data_six_seasons.csv'))
old_data = pd.read_csv(os.path.join(project_root, 'data', 'full_data.csv'))

def rookie_scale(age):
    experience = age - 20
    if 1 <= experience <= 4:
        return 1
    else:
        return 0 

full_data['rookie_scale'] = full_data['Age'].apply(rookie_scale)
old_data['rookie_scale'] = old_data['Age'].apply(rookie_scale)
test_season = 'season_2023-24'
train_data = full_data.loc[full_data[test_season] == 0].copy()
test_data = full_data.loc[full_data[test_season] == 1].copy()
features = old_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])

Lasso_rookie_reg = joblib.load(os.path.join(project_root, 'models', 'unscaled_lasso_model_with_rookie.joblib'))

feature_names = features.columns
coefficients = Lasso_rookie_reg.coef_
coef_dict = dict(zip(feature_names, coefficients))
non_zero_features = []
zero_features = []
for feature, coefficient in coef_dict.items():
    # print(f"{feature}: {coefficient}")
    if coefficient == 0.0 or coefficient == -0.0:
        zero_features.append(feature)
    else:
        non_zero_features.append(feature)
# print(f"Non Zero Features: {non_zero_features} # {len(non_zero_features)}")
X_reduced = train_data.drop(columns=zero_features + ['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap', 'season_2019-20', 'season_2020-21'])
Y_reduced = train_data['Cap_Pct']
new_X_reduced = test_data.drop(columns=zero_features + ['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap', 'season_2019-20', 'season_2020-21'])
new_Y_reduced = test_data['Cap_Pct']
print(f"X Shape: {X_reduced.shape}")
print(f"Y Shape: {Y_reduced.shape}")
print(f"new X Shape: {new_X_reduced.shape}")
print(f"new Y Shape: {new_Y_reduced.shape}")
print(f"Non Zero Variables: {X_reduced.columns} #: {len(X_reduced.columns)}")

# tunning hyperparameters and evaluating model
param_grid = {
    'n_estimators': [170, 175, 200],
    'max_depth': [None],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [4, 5, 7],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestRegressor(random_state=32)
tscv = TimeSeriesSplit(n_splits=3)
grid_seach = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_seach.fit(X_reduced, Y_reduced)
print(f"Tunning Hyperparameters via Grid Search CV")
print(f"Best Parameters: {grid_seach.best_params_}")
print(f"Best Score: {grid_seach.best_score_:.5f}")
print(f"")

# fitting model and predicting on unseen data
reduced_random_forest_mod = grid_seach.best_estimator_
new_pred = reduced_random_forest_mod.predict(new_X_reduced)
print(f"Scores for Prediction on 2023-24 Data:")
print(f"MAE: {mean_absolute_error(new_Y_reduced, new_pred):.5f}") # 0.02677
print(f"MSE: {mean_squared_error(new_Y_reduced, new_pred):.5f}") # 0.00163
print(f"")

feature_names = X_reduced.columns
feature_importance_score = reduced_random_forest_mod.feature_importances_
score_dict = dict(zip(feature_names, feature_importance_score))
feature_importance_score_df = pd.DataFrame({"feature": feature_names, "score": feature_importance_score})
feature_importance_score_df.sort_values(by='score', ascending=False, inplace=True)
print(f"Top 5 Features By Importance:")
print(f"{feature_importance_score_df.head()}")
print(f"")

model_filename = 'random_forest_reduced_model_extra_data.joblib'
# joblib.dump(reduced_random_forest_mod, os.path.join(project_root, 'models', model_filename))

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
