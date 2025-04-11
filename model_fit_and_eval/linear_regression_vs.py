import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"Linear Regression with Variable Selection from unscaled Lasso Model")
print(f"")

# loading data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full_data = pd.read_csv(os.path.join(project_root, 'data', 'full_data.csv'))
test_season = 'season_2023-24'
train_data = full_data.loc[full_data[test_season] == 0]
test_data = full_data.loc[full_data[test_season] == 1]
features = train_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])

Lasso_reg = joblib.load(os.path.join(project_root, 'models', 'unscaled_lasso_model.joblib'))
feature_names = features.columns
coefficients = Lasso_reg.coef_
coef_dict = dict(zip(feature_names, coefficients))
zero_features = []
for feature, coefficient in coef_dict.items():
    #print(f"{feature}: {coefficient:.4f}")
    if coefficient == 0.0000 or coefficient == -0.0000:
        zero_features.append(feature)
#print(zero_features, len(zero_features))

X = train_data.drop(columns=zero_features + ['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
#print(X.head(), X.shape)
Y = train_data['Cap_Pct']
new_X = test_data.drop(columns=zero_features + ['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
new_Y = test_data['Cap_Pct']

# evaluating model

tscv = TimeSeriesSplit(n_splits=3)
MAE = []
MSE = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    model = LinearRegression().fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    ab_error = mean_absolute_error(Y_test, y_pred)
    sq_error = mean_squared_error(Y_test, y_pred)
    MAE.append(ab_error)
    MSE.append(sq_error)

print(f"Model Evaluation Metrics:")
print(f"Average MAE in time series cv: {np.mean(MAE):.5f}")
print(f"Average MSE in time series cv: {np.mean(MSE):.5f}")
print(f"")

# fitting model on full data and predicting on unseen data
Lin_reg = LinearRegression().fit(X, Y)
new_pred = Lin_reg.predict(new_X)
print(f"Scores for Prediction on 2023-24 Data:")
print(f"MAE: {mean_absolute_error(new_Y, new_pred):.5f}") # 0.03392
print(f"MSE: {mean_squared_error(new_Y, new_pred):.5f}") # 0.00213
print(f"")
feature_names = X.columns
coefficients = Lin_reg.coef_
coef_dict = dict(zip(feature_names, coefficients))

model_filename = 'lin_reg_variable_selected_model.joblib'
# joblib.dump(Lin_reg, os.path.join(project_root, 'models', model_filename))

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

