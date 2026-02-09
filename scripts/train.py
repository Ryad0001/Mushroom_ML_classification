import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# 1. Chargement des données (téléchargez mushrooms.csv dans /data au préalable)
df = pd.read_csv('../data/mushrooms.csv')

# 2. Encodage des données catégorielles
# On transforme les lettres en chiffres
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('class', axis=1)
y = df['class'] # Target: 1 pour poisonous, 0 pour edible

# 3. Preprocessing : Scaler (indispensable pour la PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Embedding : PCA (Analyse en Composantes Principales)
# On réduit à 3 composantes comme dans l'exemple du TD [cite: 48]
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 5. Entraînement du modèle de prédiction
model = RandomForestClassifier(n_estimators=100)
model.fit(X_pca, y)

# 6. Création du fichier ref_data.csv [cite: 45, 46, 47]
# On structure le CSV : PCA_1, PCA_2, PCA_3, target [cite: 48]
ref_df = pd.DataFrame(X_pca, columns=['PCA 1', 'PCA 2', 'PCA 3'])
ref_df['target'] = y.astype(bool) # Conversion en booléen comme demandé [cite: 48]
ref_df['prediction'] = ref_df['target'] # Pour le rapport Evidently, on simule une prédiction parfaite sur ref_data
ref_df.to_csv('../data/ref_data.csv', index=False)

# 7. Sauvegarde des artifacts 
with open('../artifacts/model.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('../artifacts/pca.pickle', 'wb') as f:
    pickle.dump(pca, f)

with open('../artifacts/encoders.pickle', 'wb') as f:
    pickle.dump(encoders, f)

with open('../artifacts/scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

print("Setup terminé : ref_data.csv et artifacts créés !")