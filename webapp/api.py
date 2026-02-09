import streamlit as st
import requests
import pandas as pd

st.title("🍄 Mushroom Classifier")
st.write("Uploadez les caractéristiques d'un champignon pour savoir s'il est comestible.")

# 1. Interface d'upload (selon l'énoncé )
uploaded_file = st.file_uploader("Choisissez un fichier CSV contenant une observation", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Données chargées :", data)
    
    if st.button("Prédire"):
        # 2. Préparation de la requête pour l'API [cite: 71]
        # Notez l'URL spéciale : 'serving-api' est le nom du service Docker [cite: 72, 82]
        api_url = "http://serving-api:8080/predict"
        
        # On transforme la première ligne du CSV en liste pour l'API
        observation = data.values[0].tolist()
        payload = {"observation": observation}
        
        try:
            response = requests.post(api_url, json=payload)
            prediction = response.json()["prediction"]
            
            # 3. Affichage du résultat [cite: 71]
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
                    res = requests.post("http://serving-api:8080/feedback", json=feedback_payload)
                    if res.status_code == 200:
                        st.success("Merci pour votre retour ! 📝")
                    else:
                        st.error(f"Erreur lors de l'envoi du feedback: {res.text}")
                except Exception as e:
                    st.error(f"Erreur de connexion à l'API de feedback : {e}")
                
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")