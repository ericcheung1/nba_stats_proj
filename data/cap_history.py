import pandas as pd
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cap_table = pd.read_csv(os.path.join(project_root, 'data', 'cap_history.csv'))
cap_table.columns = ['season', 'Salary_cap', 'cap_in_2022']
cap_table.drop(columns=['cap_in_2022'], inplace=True)
cap_table['Salary_cap'] = cap_table['Salary_cap'].str.replace(',', '')
cap_table['Salary_cap'] = cap_table['Salary_cap'].str.replace('$', '')

cap_table.to_csv(os.path.join(project_root, 'data', 'cap_hist.csv'), index=False)