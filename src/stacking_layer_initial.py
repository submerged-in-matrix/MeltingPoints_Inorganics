from env.modules import *
from utils.repro_set_up import rng
from src.decorrelated_layer_initial import base_preds_train, base_preds_val
from data.data_split import *

# --- 5. Second-Level (“meta-learner”): Stacking Models ---
stacker = Ridge(alpha=1.0)  # Using Ridge as a meta-learner
stacker.fit(base_preds_train, y_train)
final_pred = stacker.predict(base_preds_val)