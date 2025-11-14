from env.modules import *

# --- 1. Load and Clean Data ---
df = pd.read_csv('melting_point_output.csv')  # data source: Citranation database

# Extract relevant columns
df = df.rename(columns={
    'sample/material/commonName': 'compound',
    'sample/material/condition/scalar': 'smiles',
    'sample/measurement/property/scalar/value': 'mp_C'
})
df = df[['compound', 'smiles', 'mp_C']].dropna()
df = df[df['mp_C'].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))]
df = df.reset_index(drop=True)

print(f"Number of compounds: {len(df)}")
df.head(3)