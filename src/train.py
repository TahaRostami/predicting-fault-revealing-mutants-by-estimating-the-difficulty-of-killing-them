import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

ds_name=["Codeflaws","CoREBench"][0]
params_reg={'n_estimators':None,'max_leaves':None,'learning_rate':None,'n_jobs':None,'verbosity':None}
params_clf={'n_estimators':None,'max_leaves':None,'learning_rate':None,'n_jobs':None,'verbosity':None}


X = pd.read_parquet(f"../data/preprocessed_{ds_name}_features.parquet", engine="pyarrow")
y=X['actual_class']
mutantIDs=X['mutantID']
SM_kill_freq = X['actual_SM_kill_freq']
X['actual_SM_kill_freq'] = 0
projectIDs = X['projectID']
X.drop(['projectID','mutantID','actual_class'], axis=1, inplace=True)
cv=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42)

roundx=1
for train_idxs, test_idxs in cv.split(X, y, projectIDs):
    X_train,y_train,y_train_reg=X.loc[train_idxs,:],y.loc[train_idxs],SM_kill_freq.loc[train_idxs]
    idx = ~y_train_reg.isnull()
    model_reg = xgb.XGBRegressor(**params_reg).fit(X_train.loc[idx, :], y_train_reg.loc[idx])
    model_reg.save_model(f"model_reg_{roundx}.json")
    X_train['actual_SM_kill_freq'] = model_reg.predict(X_train)
    model = xgb.XGBClassifier(**params_clf).fit(X_train, y_train)
    model.save_model(f"model_{roundx}.json")
    print(f"round {roundx}")
    roundx += 1