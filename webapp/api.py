import streamlit as st
import requests
import pandas as pd

st.title("🍄 Mushroom Classifier")
st.write("Uploadez les caractéristiques d'un champignon pour savoir s'il est comestible.")

# 1. Interface d'upload (selon l'énoncé )
uploaded_file = st.file_uploader("Choisissez un fichier CSV contenant une observation", type="csv")

if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "observations" not in st.session_state:
    st.session_state.observations = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    st.write("Données chargées :", data)
    
    # 2. Bouton Prédire
    if st.button("Prédire tout le fichier"):
        import os
        api_base_url = os.getenv("API_URL", "http://serving-api:8080")
        api_url = f"{api_base_url}/predict"
        
        predictions = []
        observations = []
        
        progress_bar = st.progress(0)
        num_rows = len(data)
        
        for i in range(num_rows):
            observation = data.iloc[i].values.tolist()
            try:
                response = requests.post(api_url, json={"observation": observation})
                if response.status_code == 200:
                    pred = response.json()["prediction"]
                    predictions.append(pred)
                    observations.append(observation)
                else:
                    predictions.append("Error")
                    observations.append(observation)
            except Exception as e:
                predictions.append(f"Connection Error: {e}")
                observations.append(observation)
            progress_bar.progress((i + 1) / num_rows)
            
        st.session_state.predictions = predictions
        st.session_state.observations = observations

    # --- AFFICHAGE PERSISTANT ---
    if st.session_state.predictions:
        preds = st.session_state.predictions
        obs_list = st.session_state.observations
        
        # Affichage sous forme de tableau
        results_df = pd.DataFrame({
            "Mushroom #": range(1, len(preds) + 1),
            "Prediction": preds
        })
        st.table(results_df)
        
        st.write("---")
        st.subheader("Feedback pour le dernier champignon du fichier")
        real_class = st.radio("Pour le dernier champignon, quelle était la classe réelle ?", ["Edible", "Poisonous"])

        if st.button("Envoyer le Feedback"):
            import os
            api_base_url = os.getenv("API_URL", "http://serving-api:8080")
            feedback_payload = {
                "observation": obs_list[-1],
                "prediction": preds[-1],
                "target": "True" if real_class == "Poisonous" else "False"
            }
            try:
                res = requests.post(f"{api_base_url}/feedback", json=feedback_payload)
                if res.status_code == 200:
                    st.success("Merci ! Feedback enregistré pour le dernier échantillon. 📝")
                else:
                    st.error(f"Erreur : {res.text}")
            except Exception as e:
                st.error(f"Erreur de connexion : {e}")
else:
    st.session_state.predictions = None
    st.session_state.observations = None