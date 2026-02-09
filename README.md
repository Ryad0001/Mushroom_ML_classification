# 🍄 Mushroom ML-Classification Project

Bienvenue sur le projet de classification de champignons, conçu dans le cadre du module "Mise en production et déploiement continu".

Ce projet implémente une **chaîne MLOps complète** :
1.  **Serving API** (FastAPI) pour les prédictions en temps réel.
2.  **Web App** (Streamlit) pour l'interface utilisateur.
3.  **Monitoring** (Evidently) pour surveiller la dérive des données (Data Drift) et la performance.
4.  **Continuous Deployment** avec ré-entraînement automatique.

---

## 🚀 Comment Lancer le Projet (Quickstart)

### Pré-requis
- Docker Desktop installé et lancé.
- Git.

### Étape 1 : Récupérer le code
```bash
git clone <VOTRE_REPO_GIT>
cd ML-classification
```

### Étape 2 : Lancer les services
Exécutez simplement les commandes suivantes dans votre terminal :

```bash
# 1. Lancer l'API de prédiction
docker compose -f serving/docker-compose.yml up --build -d

# 2. Lancer l'Application Web
docker compose -f webapp/docker-compose.yml up --build -d

# 3. Lancer le système de Monitoring
docker compose -f reporting/docker-compose.yml up --build -d
```

> **Note Mac/Linux** : Si vous avez des erreurs de permission, ajoutez `DOCKER_BUILDKIT=0` devant les commandes `docker compose`.

---

## 🔗 Accès aux Interfaces

Une fois les conteneurs lancés :

| Service | URL | Description |
| :--- | :--- | :--- |
| **Web App** | [http://localhost:8081](http://localhost:8081) | Uploadez un CSV (`data/test_samples.csv` fourni) pour tester. |
| **Monitoring** | [http://localhost:8082](http://localhost:8082) | Tableau de bord Evidently (Data Drift, Accuracy...). Allez dans l'onglet **Reports**. |
| **API Docs** | [http://localhost:8080/docs](http://localhost:8080/docs) | Documentation Swagger de l'API. |

---

## ✅ Conformité au Sujet (Technical Documentation)

Ce projet respecte scrupuleusement les 5 points du cahier des charges :

### I. Préparation des Données
- Utilisation de `StandardScaler` et `PCA` (3 composantes conservées).
- Séparation stricte Train/Test.

### II. API de Serving (FastAPI)
- Endpoint `/predict` fonctionnel.
- Chargement des modèles (`model.pickle`, `pca.pickle`, etc.) au démarrage via `@app.on_event("startup")` (Variables Globales).

### III. Interface Web (Streamlit)
- Upload de fichier CSV supporté.
- **Batch Testing** : Prédiction possible sur tout un fichier d'un coup.
- **Feedback Loop** : Interface permettant de corriger les prédictions et d'envoyer la vérité terrain à l'API.

### IV. Reporting (Evidently)
- Calcul du **Data Drift** sur les nouvelles données de production.
- Métriques de Classification : **F1-Score, Accuracy, Recall, Precision**.
- Les rapports sont générés et stockés dans un **Workspace Evidently** persistent.

### V. Ré-entraînement Continu (CD)
- Endpoint `/feedback` qui enregistre les données dans `prod_data.csv`.
- **Seuil K=10** : Le ré-entraînement se déclenche automatiquement tous les 10 feedbacks.
- **Hot-Swap** : Le modèle en production est mis à jour "à chaud" en mémoire sans interruption de service.

---

## � Structure du Projet
```
.
├── artifacts/          # Modèles entraînés (.pickle)
├── data/               # Données (mushrooms.csv, ref_data.csv, prod_data.csv)
├── serving/            # API FastAPI + Dockerfile
├── webapp/             # Application Streamlit + Dockerfile
├── reporting/          # Script Evidently + Dockerfile
├── scripts/            # Scripts de training et génération de données
└── docker-compose.yml  # (Découpé en 3 fichiers spécifiques par dossier)
```
