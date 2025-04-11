import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"Lasso Regression with Scaled Features")
print(f"")

# loading data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full_data = pd.read_csv(os.path.join(project_root, 'data', 'full_data.csv'))
test_season = 'season_2023-24'
train_data = full_data.loc[full_data[test_season] == 0]
test_data = full_data.loc[full_data[test_season] == 1]
X = train_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
Y = train_data['Cap_Pct']
new_X = test_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
new_Y = test_data['Cap_Pct']
numerical_cols = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG_PCT', '_3P', '_3PA', '_3P_PCT',
       '_2P', '_2PA', '_2P_PCT', 'eFG_PCT', 'FT', 'FTA', 'FT_PCT', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'TMP', 'PER',
       'TS_PCT', '_3PAr', 'FTr', 'ORB_PCT', 'DRB_PCT', 'TRB_PCT', 'AST_PCT',
       'STL_PCT', 'BLK_PCT', 'TOV_PCT', 'USG_PCT', 'OWS', 'DWS', 'WS',
       'WSPER48', 'OBPM', 'DBPM', 'BPM', 'VORP']

# evaluating data and parameter tunning
alphas_to_test = np.logspace(-4, 1, 15)
tscv = TimeSeriesSplit(n_splits=3)
all_MAE = []
all_MSE = []
best_alpha = None
best_averaged_MAE = float('inf')

for alpha in alphas_to_test: # loop through each alpha to be tested
    MAE_for_alpha = []
    MSE_for_alpha = []
    for train_index, test_index in tscv.split(X): # cross validating for current alpha
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        scaler = StandardScaler()
        X_train_numerical = X_train[numerical_cols]
        X_test_numerical = X_test[numerical_cols]
        X_train_scaled = scaler.fit_transform(X_train_numerical)
        X_test_scaled = scaler.transform(X_test_numerical)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train_numerical.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test_numerical.columns)

        X_train_processed = pd.concat([X_train_scaled_df, X_train.drop(columns=numerical_cols, axis=1)], axis=1)
        X_test_processed = pd.concat([X_test_scaled_df, X_test.drop(columns=numerical_cols, axis=1)], axis=1)

        model = Lasso(alpha, random_state=32, max_iter=25000).fit(X_train_processed, Y_train)
        y_pred = model.predict(X_test_processed)
        ab_error = mean_absolute_error(Y_test, y_pred)
        sq_error = mean_squared_error(Y_test, y_pred)
        #print(f"MAE for CV: {ab_error}")
        MAE_for_alpha.append(ab_error) # scores for current cv fold
        MSE_for_alpha.append(sq_error)
    
    #print(f"MAEs for Alpha: {MAE_for_alpha}")
    averaged_MAE = np.mean(MAE_for_alpha) # scores for current alpha
    #print(f"Averaged MAEs for Alpha: {averaged_MAE}")
    averaged_MSE = np.mean(MSE_for_alpha)
    all_MAE.append(averaged_MAE) # addes average cv score to overall list
    all_MSE.append(averaged_MSE)


    if averaged_MAE < best_averaged_MAE: # if current alpha score is lowest
        best_averaged_MAE = averaged_MAE 
        best_alpha = alpha

print(f"Model Evaluation Metrics:")
print(f"Best alpha through time series cv is: {best_alpha:.6f}")
print(f"Best MAE found through time series: {best_averaged_MAE:.6f}")
print(f"")

# fitting model on full data and predicting on unseen data
scaler = StandardScaler()
X_num = X[numerical_cols]
X_scaled = scaler.fit_transform(X_num)
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X_num.columns)
X_processed = pd.concat([X_scaled_df, X.drop(columns=numerical_cols, axis=1)], axis=1)

Lasso_reg = Lasso(alpha=0.0011787686347935866, random_state=32, max_iter=25000).fit(X_processed, Y)
new_X_num = new_X[numerical_cols]
new_X_scaled = scaler.transform(new_X_num)
new_X_scaled_df = pd.DataFrame(new_X_scaled, index=new_X.index, columns=new_X_num.columns)
new_X_processed = pd.concat([new_X_scaled_df, new_X.drop(columns=numerical_cols, axis=1)], axis=1)
new_pred = Lasso_reg.predict(new_X_processed)
print(f"Scores for Prediction on 2023-24 Data:")
print(f"MAE: {mean_absolute_error(new_Y, new_pred):.5f}") # 0.03401
print(f"MSE: {mean_squared_error(new_Y, new_pred):.5f}") # 0.00211
print(f"")
feature_names = X.columns
coefficients = Lasso_reg.coef_
coef_dict = dict(zip(feature_names, coefficients))

# for feature, coefficient in coef_dict.items():
#     print(f"{feature}: {coefficient:.4f}")

model_filename = 'scaled_lasso_model.joblib'
# joblib.dump(Lasso_reg, os.path.join(project_root, 'models', model_filename))

# final predictions on new unseen data
new_pred_series = pd.Series(new_pred)
test_data_copy = test_data.copy()
test_data_copy.reset_index(inplace=True)
test_data_copy['Predicted_Cap_Pct'] = np.round(new_pred_series, 4)
test_data_copy['Predicted_Salary'] = np.round(test_data_copy['Predicted_Cap_Pct']*test_data_copy['Salary_cap'], 0)
final_preds = test_data_copy[['player', 'Cap_Pct', 'Predicted_Cap_Pct', 'Salary', 'Predicted_Salary']].copy()
final_preds[['Salary', 'Predicted_Salary']] = final_preds[['Salary', 'Predicted_Salary']].map(lambda x: f"${x:,.0f}")
print(f"Salary Predictions:")
print(f"{final_preds.sample(5)}")
