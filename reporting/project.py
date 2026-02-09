import pandas as pd
import numpy as np
import os
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.ui.workspace import Workspace
from evidently.pipeline.column_mapping import ColumnMapping

# Expert MLOps: Strict TD Compliance (Evidently v0.4.0 Implementation)
def generate_report():
    try:
        print("--- Mushroom Monitoring Report Generation ---")
        
        # Paths for TD compliance
        REF_PATH = '/data/ref_data.csv'
        PROD_PATH = '/data/prod_data.csv'
        
        if not os.path.exists(REF_PATH) or not os.path.exists(PROD_PATH):
            print("Erreur : Fichiers ref_data.csv ou prod_data.csv manquants.")
            return

        ref_data = pd.read_csv(REF_PATH)
        prod_data = pd.read_csv(PROD_PATH)

        # Harmonisation des types pour la target (Expert: essential for metrics)
        map_dict = {'True': True, 'False': False, True: True, False: False, 1: True, 0: False}
        if 'target' in ref_data.columns:
            ref_data['target'] = ref_data['target'].map(map_dict)
        if 'target' in prod_data.columns:
            prod_data['target'] = prod_data['target'].map(map_dict)
        
        # TD REQUIREMENT: Use ColumnMapping (Exact syntax for v0.4.0)
        column_mapping = ColumnMapping()
        column_mapping.target = 'target'
        column_mapping.prediction = 'prediction'
        column_mapping.numerical_features = ['PCA 1', 'PCA 2', 'PCA 3']
        
        # 1. Génération du Rapport
        # TD REQUIREMENT: Inclus le Data Drift et les métriques de classification (F1, Accuracy, Recall, Precision)
        # Expert MLOps: Use robust metrics that work well with label-only data
        from evidently.metrics import ClassificationQualityMetric, DatasetDriftMetric, ColumnDriftMetric
        report = Report(metrics=[
            ClassificationQualityMetric(),  # Inclus Accuracy, Precision, Recall, F1
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name="target"),
        ])

        print("Calcul des métriques (Classification et Drift)...")
        # Exact syntax for v0.4.0
        report.run(reference_data=ref_data, current_data=prod_data, column_mapping=column_mapping)

        # 2. Sauvegarde dans le Workspace (TD REQUIREMENT: use add_run and as_snapshot)
        workspace = Workspace("workspace")
        projects = workspace.list_projects()
        project = next((p for p in projects if p.name == "Mushroom Monitoring"), None)
        if project is None:
            project = workspace.create_project("Mushroom Monitoring")
        
        if hasattr(report, "as_snapshot"):
            snapshot = report.as_snapshot()
        else:
            snapshot = report.to_snapshot()

        workspace.add_run(project.id, snapshot)
        print("Rapport généré avec succès dans le workspace.")

    except Exception as e:
        print(f"Erreur lors de la génération du rapport : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_report()
