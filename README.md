# Mushroom Classification - MLOps Project

This project implements a complete end-to-end machine learning pipeline for Mushroom classification (Edible vs Poisonous), following strict technical documentation requirements.

## 🏗️ Architecture

The project is orchestrated using **Docker Compose**, splitting the system into three independent services communicating over a shared network (`serving_prod_net`).

*   **Serving API (FastAPI)**: Port `8080`. Manages model inference and the feedback loop.
*   **Web Application (Streamlit)**: Port `8081`. User interface for predictions and feedback submission.
*   **Reporting (Evidently)**: Port `8082`. Monitoring dashboard for Data Drift and Classification metrics (F1, Accuracy, Precision, Recall).

## 📂 Project Structure

```text
.
├── artifacts/          # ML Models (pickle), Scalers, and Embeddings
├── data/               # Datasets (ref_data.csv, prod_data.csv)
├── scripts/            # Training scripts and EDA notebooks
├── serving/            # FastAPI code, Dockerfile, and Compose
├── webapp/             # Streamlit code, Dockerfile, and Compose
├── reporting/          # Evidently monitoring script, UI, and Compose
└── README.md           # Project documentation
```

## 🚀 Getting Started

Ensure you have Docker and Docker Compose installed.

### 1. Launch Services
Run the following commands from the root directory:

```bash
# Start the Serving API (Creates the network)
docker compose -f serving/docker-compose.yml up --build -d

# Start the Webapp
docker compose -f webapp/docker-compose.yml up --build -d

# Start the Monitoring
docker compose -f reporting/docker-compose.yml up --build -d
```

### 2. Access the Applications
*   **Web Interface**: [http://localhost:8081](http://localhost:8081)
*   **API Documentation**: [http://localhost:8080/docs](http://localhost:8080/docs)
*   **Monitoring Dashboard**: [http://localhost:8082](http://localhost:8082)

## 🔄 Continuous Deployment

### Model Re-training Trigger
The system includes an automated re-training loop triggered by user feedback:
1.  Users provide feedback via the web app (Target vs Prediction).
2.  Data is embedded (PCA) and appended to `prod_data.csv`.
3.  Every **k=10** new entries, the API triggers a re-training on `ref_data + prod_data`.
4.  The API **Hot-Swaps** the global `MODEL` variable in memory for zero-downtime updates.

## 📊 Monitoring Metrics

The `reporting` service generates Evidently snapshots including:
*   **Classification Quality**: F1-Score, Precision, Recall, Balanced Accuracy.
*   **Data Drift**: Statistical drift detection on PCA features and target.

To manually trigger a report update:
```bash
docker exec reporting python project.py
```

## 🛠️ Requirements & Dependencies
*   Python 3.10
*   FastAPI & Uvicorn
*   Streamlit
*   Scikit-learn
*   Evidently (v0.5.0)
*   Pandas & NumPy
