from env.modules import *
from featurize.featurize_custom import X, y
from data.data_cleaning import df

# --- 3. Train-Test Split ---
X_temp, X_unseen, y_temp, y_unseen = train_test_split(
    X, y, test_size=141, random_state=42
)

# Then split the rest into train/validation as usual (e.g., 80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=400, random_state=42
)

print(f"Train samples: {len(y_train)}, Validation samples: {len(y_val)}, Unseen samples: {len(y_unseen)}")

# compound_names_all = df['compound']  # get compound names only
# _, compound_unseen = train_test_split(
#     compound_names_all, test_size=141, random_state=42)
