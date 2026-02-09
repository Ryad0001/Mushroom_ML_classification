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
        st.subheader("Feedback en masse")
        st.write("Soumettez les labels réels pour toutes les observations ci-dessus.")
        bulk_class = st.radio("Sont-ils tous ?", ["Edible", "Poisonous"], key="bulk_radio")

        if st.button("Envoyer le Feedback pour TOUT le fichier"):
            import os
            api_base_url = os.getenv("API_URL", "http://serving-api:8080")
            success_count = 0
            error_count = 0
            
            progress_bulk = st.progress(0)
            total = len(obs_list)
            
            for i in range(total):
                feedback_payload = {
                    "observation": obs_list[i],
                    "prediction": preds[i],
                    "target": "True" if bulk_class == "Poisonous" else "False"
                }
                try:
                    res = requests.post(f"{api_base_url}/feedback", json=feedback_payload)
                    if res.status_code == 200:
                        success_count += 1
                    else:
                        error_count += 1
                except:
                    error_count += 1
                progress_bulk.progress((i + 1) / total)
            
            st.success(f"Terminé ! {success_count} feedbacks envoyés avec succès. ✅")
            if error_count > 0:
                st.warning(f"{error_count} erreurs rencontrées.")
            
            # Message spécial si le ré-entraînement a dû être déclenché
            if success_count >= 10:
                st.info("💡 Note : Le seuil de 10 ayant été atteint, le modèle a été ré-entraîné automatiquement.")
else:
    st.session_state.predictions = None
    st.session_state.observations = None