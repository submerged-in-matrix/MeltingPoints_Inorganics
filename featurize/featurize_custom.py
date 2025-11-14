from env.modules import *
from data.data_cleaning import df
from utils.feat_helpers import *

# Featurization: SMILES to RDKit Descriptors + Morgan Fingerprints + Function to count bond types in a molecule
# Feature and target extraction

feature_labels = [
    'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumDoubleBonds', 'NumTripleBonds', 'NumAromaticBonds', 'FractionCSP3'
] + [f'FP_{i}' for i in range(128)]

# Apply featurizer
feats = df['smiles'].apply(featurize_smiles)
X = np.vstack(feats)
X = pd.DataFrame(X, columns=feature_labels)
X = X.fillna(X.median())                    # handle parsing errors
X.to_csv('features.csv', index=False)  # save features for future use
y = df['mp_C'].astype(float)
y.to_csv('targets.csv', index=False)  # save targets for future use