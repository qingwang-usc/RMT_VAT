from sklearn.externals import joblib
import pandas as pd
study = joblib.load('/home/lindacjx/tcsm/skin_optuna_50.pkl')
df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete'], axis=1)
df.to_csv('/home/lindacjx/tcsm/skin.csv')
columns = df.columns.values.tolist()
print(df.head(30))# !pip install optuna
print(columns)
#print(df['params_alpha'])
#print(df['params_epochs'])
print(study.best_trial)