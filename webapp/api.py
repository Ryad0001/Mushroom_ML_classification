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
        
        st.write("---")
        st.subheader("Validation des Résultats")
        st.write("Vérifiez les prédictions et corrigez la colonne 'True Class' si nécessaire.")

        # Préparation du DataFrame éditable
        # On pré-remplit "True Class" avec la prédiction (l'utilisateur corrige juste les erreurs)
        df_init = pd.DataFrame({
            "Mushroom Index": range(len(preds)),
            "Prediction": preds,
            "True Class": preds # Par défaut, on suppose que le modèle a raison
        })

        # Widget d'édition interactif
        edited_df = st.data_editor(
            df_init,
            column_config={
                "Mushroom Index": st.column_config.NumberColumn(disabled=True),
                "Prediction": st.column_config.TextColumn(disabled=True),
                "True Class": st.column_config.SelectboxColumn(
                    "Label Réel",
                    options=["Edible", "Poisonous"],
                    help="Sélectionnez la classe réelle observée",
                    required=True
                )
            },
            disabled=["Mushroom Index", "Prediction"],
            hide_index=True,
            use_container_width=True
        )

        if st.button("Valider et Envoyer les Feedbacks"):
            import os
            api_base_url = os.getenv("API_URL", "http://serving-api:8080")
            success_count = 0
            error_count = 0
            
            progress_bulk = st.progress(0)
            total = len(preds)
            
            # On itère sur le DataFrame édité
            for i, row in edited_df.iterrows():
                # On récupère l'index d'origine pour retrouver l'observation correspondante
                idx = row["Mushroom Index"]
                true_label = row["True Class"]
                
                feedback_payload = {
                    "observation": obs_list[idx],
                    "prediction": preds[idx],
                    "target": "True" if true_label == "Poisonous" else "False"
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
            
            st.success(f"Traitement terminé ! {success_count} feedbacks envoyés. ✅")
            if error_count > 0:
                st.warning(f"{error_count} erreurs pendant l'envoi.")
            
            if success_count >= 10:
                st.balloons()
                st.info("🎯 Seuil de ré-entraînement atteint ! Le modèle s'est mis à jour.")
else:
    st.session_state.predictions = None
    st.session_state.observations = None