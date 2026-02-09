import streamlit as st
import requests
import pandas as pd

st.title("🍄 Mushroom Classifier")
st.write("Uploadez les caractéristiques d'un champignon pour savoir s'il est comestible.")

# 1. Interface d'upload (selon l'énoncé )
uploaded_file = st.file_uploader("Choisissez un fichier CSV contenant une observation", type="csv")

# --- INITIALISATION DU STATE ---
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "observation" not in st.session_state:
    st.session_state.observation = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Données chargées :", data)
    
    # 2. Bouton Prédire
    if st.button("Prédire"):
        import os
        api_base_url = os.getenv("API_URL", "http://serving-api:8080")
        api_url = f"{api_base_url}/predict"
        
        # On transforme la première ligne du CSV en liste pour l'API
        observation = data.values[0].tolist()
        payload = {"observation": observation}
        
        try:
            response = requests.post(api_url, json=payload)
            # On stocke dans le session_state pour persister au rerun
            st.session_state.prediction = response.json()["prediction"]
            st.session_state.observation = observation
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")

    # --- AFFICHAGE PERSISTANT DU RÉSULTAT ET FEEDBACK ---
    if st.session_state.prediction:
        prediction = st.session_state.prediction
        observation = st.session_state.observation
        import os
        api_base_url = os.getenv("API_URL", "http://serving-api:8080")
        
        # 3. Affichage du résultat
        if prediction == "Edible":
            st.success(f"Résultat : {prediction} ✅")
        else:
            st.error(f"Résultat : {prediction} ⚠️")
        
        # --- FEEDBACK UI ---
        st.write("---")
        st.subheader("Aidez-nous à améliorer le modèle")
        real_class = st.radio("Quelle était la classe réelle ?", ["Edible", "Poisonous"])

        if st.button("Envoyer le Feedback"):
            feedback_payload = {
                "observation": observation,
                "prediction": prediction,
                "target": "True" if real_class == "Poisonous" else "False" # Target logic as requested: Poisonous=True, Edible=False
            }
            try:
                res = requests.post(f"{api_base_url}/feedback", json=feedback_payload)
                if res.status_code == 200:
                    st.success("Merci pour votre retour ! 📝")
                else:
                    st.error(f"Erreur lors de l'envoi du feedback: {res.text}")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API de feedback : {e}")
else:
    # Reset si le fichier est enlevé
    st.session_state.prediction = None
    st.session_state.observation = None