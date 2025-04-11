import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"Lasso Regression with Unscaled Features and Rookie Scale Contract Feature")
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
train_data = full_data.loc[full_data[test_season] == 0]
test_data = full_data.loc[full_data[test_season] == 1]
X = train_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
Y = train_data['Cap_Pct']
new_X = test_data.drop(columns=['Cap_Pct', 'player', 'player_id', 'Salary', 'Salary_cap'])
new_Y = test_data['Cap_Pct']

# evaluating model and parameter tunning
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

        model = Lasso(alpha, random_state=32, max_iter=25000).fit(X_train, Y_train)
        y_pred = model.predict(X_test)
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
print(f"Best MAE found through time series cv is: {best_averaged_MAE:.6f}")
print(f"")

# fitting model on full data and predicting on unseen data
Lasso_reg = Lasso(alpha=0.0011787686347935866, random_state=32, max_iter=25000).fit(X, Y)
new_pred = Lasso_reg.predict(new_X)
print(f"Scores for Prediction on 2023-24 Data:")
print(f"MAE: {mean_absolute_error(new_Y, new_pred):.5f}") 
print(f"MSE: {mean_squared_error(new_Y, new_pred):.5f}") 
print(f"")
feature_names = X.columns
coefficients = Lasso_reg.coef_
coef_dict = dict(zip(feature_names, coefficients))

non_zero = []
for feature, coefficient in coef_dict.items():
    if coefficient != 0.0000 or coefficient != -0.0000:
        non_zero.append(feature)
        #print(f"{feature}: {coefficient:.4f}")
print(f"Non-zero features: {non_zero} Number: {len(non_zero)}")

model_filename = 'unscaled_lasso_model_with_rookie.joblib'
# joblib.dump(Lasso_reg, os.path.join(project_root, 'models', model_filename))

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