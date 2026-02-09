import pandas as pd
import pickle
import os

# Ce script génère un fichier de test avec plusieurs champignons encodés
# pour faciliter les tests en vrac (bulk testing) dans la Webapp.

# 1. Charger les données originales et les encodeurs
df = pd.read_csv('data/mushrooms.csv')
with open('artifacts/encoders.pickle', 'rb') as f:
    encoders = pickle.load(f)

# 2. Prendre un échantillon de 20 champignons
sample_df = df.sample(20, random_state=42)
true_classes = sample_df['class'].tolist()

# 3. Encoder l'échantillon
encoded_samples = []
for _, row in sample_df.iterrows():
    encoded_row = []
    for col in df.columns:
        if col != 'class':
            encoded_val = encoders[col].transform([row[col]])[0]
            encoded_row.append(encoded_val)
    encoded_samples.append(encoded_row)

# 4. Créer le CSV de test (sans la colonne class pour simuler une donnée inconnue)
# On nomme les colonnes observation_0, observation_1, etc. ou juste une liste
test_df = pd.DataFrame(encoded_samples)
test_df.to_csv('data/test_samples.csv', index=False, header=False)

print(f"Fichier data/test_samples.csv créé avec 20 champignons.")
print(f"Classes réelles pour vérification : {true_classes}")
