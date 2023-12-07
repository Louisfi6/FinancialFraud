import pandas as pd

# Charger le dataset clean.csv
df = pd.read_csv('clean.csv')

# Réduire de moitié le dataset
df_reduit = df.sample(frac=0.5, random_state=42)

# Enregistrer le nouveau dataset réduit en CSV
df_reduit.to_csv('clean_reduit.csv', index=False)
