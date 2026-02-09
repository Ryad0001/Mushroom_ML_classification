# Rapport de Réalisation - Mushroom ML-Classification

Ce document détaille l'ensemble des tâches techniques accomplies pour répondre aux exigences du projet.

## 🛠️ I. Préparation & Données
- [x] Script de vectorisation via **PCA** (3 composantes).
- [x] Création et normalisation du fichier `ref_data.csv`.
- [x] Entraînement initial du modèle et sauvegarde des artifacts (`model.pickle`, `pca.pickle`, `scaler.pickle`).

## 🚀 II. API de Serving (FastAPI)
- [x] Implémentation du fichier `serving/api.py`.
- [x] Création de l'Endpoint `/predict` (POST) avec pipeline de prétraitement intégré.
- [x] Chargement des modèles en **variables globales** au démarrage (`@app.on_event("startup")`).
- [x] Dockerisation complète de l'API avec isolation des dépendances (`requirements.txt`).

## 🌐 III. Interface Web (Streamlit)
- [x] Développement de `webapp/api.py` pour l'upload et la prédiction.
- [x] Communication inter-conteneurs via le service Docker `serving-api`.
- [x] **Correctif State Management** : Intégration de `st.session_state` pour empêcher la disparition de l'UI lors des interactions de feedback.
- [x] Dockerisation et intégration au réseau `serving_prod_net`.

## 📊 IV. Reporting (Evidently)
- [x] Implémentation de `reporting/project.py`.
- [x] Configuration des métriques de **Data Drift** et de **Classification** (F1, Accuracy, Recall, Precision).
- [x] Automatisation de la génération de snapshots dans un **Workspace**.
- [x] Dockerisation du dashboard sur le port `8082`.
- [x] Résolution des erreurs de calcul (NumPy/Evidently) via l'encodage entier des labels.

## 🔄 V. Ré-entraînement Continu
- [x] Implémentation de l'Endpoint `/feedback` (POST).
- [x] Enregistrement persistant dans `data/prod_data.csv`.
- [x] Déclenchement automatique du ré-entraînement au seuil **k=10**.
- [x] Mécanisme de **Hot-Swap** : Mise à jour du modèle en mémoire sans redémarrer le serveur API.

## 🕸️ Infrastructure & Réseau
- [x] Orchestration via **Docker Compose** multi-fichiers.
- [x] Mise en place du réseau de bridge `serving_prod_net`.
- [x] Persistence des volumes `data/` et `artifacts/` partagés entre containers.
