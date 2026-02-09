from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# --- VARIABLES GLOBALES ---
# On les définit ici pour qu'elles soient accessibles par toutes les fonctions
MODEL = None
SCALER = None
PCA_MODEL = None

@app.on_event("startup")
def load_artifacts():
    """Charge les modèles au lancement pour éviter de les relire à chaque prédiction."""
    global MODEL, SCALER, PCA_MODEL
    # Les chemins utilisent '/artifacts/' car ce sera le chemin INTERNE au conteneur Docker
    with open("/artifacts/model.pickle", "rb") as f:
        MODEL = pickle.load(f)
    with open("/artifacts/scaler.pickle", "rb") as f:
        SCALER = pickle.load(f)
    with open("/artifacts/pca.pickle", "rb") as f:
        PCA_MODEL = pickle.load(f)
    print("Modèles chargés avec succès !")

@app.post("/predict")
async def predict(data: dict):
    """
    Endpoint de prédiction. 
    Reçoit un JSON, transforme la donnée et renvoie la prédiction.
    """
    # 1. Extraction des données du JSON
    # 'observation' doit contenir la liste des caractéristiques du champignon
    features = np.array(data["observation"]).reshape(1, -1)
    
    # 2. Pipeline de traitement (Preprocessing + Embedding)
    # On doit traiter la donnée exactement comme pendant l'entraînement [cite: 58]
    features_scaled = SCALER.transform(features)
    features_pca = PCA_MODEL.transform(features_scaled)
    
    # 3. Prédiction
    prediction = MODEL.predict(features_pca)
    result = "Poisonous" if prediction[0] == 1 else "Edible"
    
    return {"prediction": result}

K_THRESHOLD = 10 # Seuil de ré-entraînement

def retrain_model_on_all_data():
    """Fusionne ref_data et prod_data pour ré-entraîner le modèle."""
    print("Déclenchement du ré-entraînement automatique...")
    # 1. Charger les deux jeux de données
    ref_df = pd.read_csv("/data/ref_data.csv")
    prod_df = pd.read_csv("/data/prod_data.csv")
    
    # 2. Fusionner
    # On garde les colonnes PCA et target pour l'entraînement
    features_cols = ["PCA 1", "PCA 2", "PCA 3"]
    target_col = "target"
    
    # Ensure consistent types for target (boolean)
    map_dict = {'True': True, 'False': False, True: True, False: False, 1: True, 0: False}
    ref_df[target_col] = ref_df[target_col].map(map_dict)
    prod_df[target_col] = prod_df[target_col].map(map_dict)
    
    X_ref = ref_df[features_cols]
    y_ref = ref_df[target_col]
    
    X_prod = prod_df[features_cols]
    y_prod = prod_df[target_col]
    
    X_all = pd.concat([X_ref, X_prod], ignore_index=True)
    y_all = pd.concat([y_ref, y_prod], ignore_index=True)
    
    # 3. Ré-entraîner
    new_model = RandomForestClassifier(n_estimators=100)
    new_model.fit(X_all, y_all)
    
    # 4. Sauvegarder l'artifact
    with open("/artifacts/model.pickle", "wb") as f:
        pickle.dump(new_model, f)
        
    print(f"Modèle ré-entraîné sur {len(X_all)} observations et sauvegardé dans /artifacts/model.pickle")
    return new_model

@app.post("/feedback")
async def feedback(data: dict):
    global MODEL, PCA_MODEL, SCALER
    
    try:
        # 1. Extraire les données
        observation = np.array(data["observation"]).reshape(1, -1)
        prediction = data["prediction"] # "Poisonous" ou "Edible"
        target = data["target"]         # "True" (Poison) ou "False" (Edible)
        
        # 2. Transformer l'observation en vecteurs PCA
        features_scaled = SCALER.transform(observation)
        features_pca = PCA_MODEL.transform(features_scaled)
        
        # 3. Préparer la ligne pour prod_data.csv
        new_row = {
            "PCA 1": features_pca[0][0],
            "PCA 2": features_pca[0][1],
            "PCA 3": features_pca[0][2],
            "target": target,
            "prediction": True if prediction == "Poisonous" else False
        }
        
        # 4. Sauvegarder dans /data/prod_data.csv
        df_prod_row = pd.DataFrame([new_row])
        # Check if file exists to decide on header
        import os
        header = not os.path.exists("/data/prod_data.csv")
        df_prod_row.to_csv("/data/prod_data.csv", mode='a', header=header, index=False)
        
        # 5. Vérification du seuil K=10
        df_prod = pd.read_csv("/data/prod_data.csv")
        retrain_triggered = False
        
        if len(df_prod) % K_THRESHOLD == 0:
            # Déclenchement du ré-entraînement
            MODEL = retrain_model_on_all_data()
            retrain_triggered = True
            
        return {
            "status": "Feedback enregistré", 
            "retrain_triggered": retrain_triggered,
            "total_feedback": len(df_prod)
        }
        
    except Exception as e:
        print(f"Erreur endpoint feedback: {e}")
        return {"error": str(e)}