import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    

print(f"Best alpha through time series cv is: {best_alpha:.6f}")
print(f"Best MAE found through time series: {best_averaged_MAE:.6f}")

# fitting model on full data
Lasso_reg = Lasso(alpha=0.0011787686347935866, random_state=32, max_iter=25000).fit(X, Y)
feature_names = X.columns
coefficients = Lasso_reg.coef_
coef_dict = dict(zip(feature_names, coefficients))

# for feature, coefficient in coef_dict.items():
#     print(f"{feature}: {coefficient:.4f}")

model_filename = 'unscaled_lasso_model.joblib'
joblib.dump(Lasso_reg, os.path.join(project_root, 'models', model_filename))

# predicting on unseen data
new_pred = Lasso_reg.predict(new_X)
print(f"MAE: {mean_absolute_error(new_Y, new_pred):.5f}")
print(f"MSE: {mean_squared_error(new_Y, new_pred):.5f}")


