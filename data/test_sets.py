import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# full test set
full_data = pd.read_csv(os.path.join(project_root, 'data', 'full_data.csv'))
test_season = 'season_2023-24'
test_data = full_data.loc[full_data[test_season] == 1].copy()
full_test = test_data.drop(columns=['player_id', 'Salary_cap'])
print(f"X Columns: {full_test.columns}, # {len(full_test.columns)}")
# print(f"{full_test.head()}")

# scaled test set
col_name = full_test.columns
encoded = ['Team_', 'season_', 'Pos_', 'player', 'Cap_', 'Salary']
num_cols = [col for col in col_name if not any(prefix in col for prefix in encoded)]
scaled_num_cols = StandardScaler().fit_transform(full_test[num_cols].copy())
scaled_df = pd.DataFrame(scaled_num_cols, index=full_test.index, columns=full_test[num_cols].copy().columns)
full_test_scaled = pd.concat([scaled_df, full_test.drop(columns=num_cols, axis=1)], axis=1)
# print(f"numerical: {num_cols}")
# print(f"new x scaled:")
# print(f"X Columsn: {full_test_scaled.columns}, # {len(full_test_scaled.columns)}")
# print(full_test_scaled.head())

# reduced without rookie
LASSO_unscaled = joblib.load(os.path.join(project_root, 'models', 'unscaled_lasso_model.joblib'))
coefs = LASSO_unscaled.coef_
col_name = full_test.drop(columns=['player', 'Cap_Pct', 'Salary']).copy().columns
coef_dict = dict(zip(col_name, coefs))
zeroed_features = []
for feature, coefficient in coef_dict.items():
    # print(f"{feature}: {coefficient:.4f}")
    if coefficient == 0.0 or coefficient == -0.0:
        zeroed_features.append(feature)
# print(zeroed_features, len(zeroed_features))
reduced_test = full_test.drop(columns=zeroed_features)
# print(f"Columns: {reduced_test.columns}, # {len(reduced_test.columns)}")
# print(reduced_test.head())

# full test set + rookie
def rookie_scale(age):
    experience = age - 20
    if 1 <= experience <= 4:
        return 1
    else:
        return 0 

full_test_rookie = full_data.loc[full_data[test_season] == 1].copy()
full_test_rookie.drop(columns=['player_id', 'Salary_cap'], inplace=True)
full_test_rookie['rookie_scale'] = test_data['Age'].apply(rookie_scale)
print(f"X Columns: {full_test_rookie.columns}, # {len(full_test_rookie.columns)}")

# reduced test set + rookie 
reduced_test_rookie = full_test_rookie.drop(columns=zeroed_features)
# print(reduced_test_rookie.columns, len(reduced_test_rookie.columns))
# print(reduced_test_rookie.head())