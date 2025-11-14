from data.data_split import X_train, X_val, X_unseen
from featurize.feat_importance import sorted_feature_names

# Select top 68 features by SHAP ranking
top_n = 68
selected_features = list(sorted_feature_names[:top_n])

X_train_sel = X_train[selected_features]
X_val_sel = X_val[selected_features]
X_unseen_sel = X_unseen[selected_features]