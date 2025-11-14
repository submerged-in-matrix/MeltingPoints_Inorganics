from env.modules import *
from data.data_opt_feats import X_train_sel, X_val_sel, X_unseen_sel

## use this non-df version for tuning MLP with BayessearchCV
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_sel)
# X_val_scaled = scaler.transform(X_val_sel)


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=X_train_sel.columns, index=X_train_sel.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_sel), columns=X_val_sel.columns, index=X_val_sel.index)
X_test_final = pd.DataFrame(scaler.fit_transform(X_unseen_sel), columns=X_unseen_sel.columns, index=X_unseen_sel.index)