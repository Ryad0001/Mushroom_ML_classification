import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from evidently.ui.workspace import Workspace
from evidently.core.datasets import Dataset
from evidently.legacy.utils.data_preprocessing import create_data_definition

# Bridge for ColumnMapping (Version compatibility)
try:
    from evidently.utils import ColumnMapping
except ImportError:
    from evidently.legacy.pipeline.column_mapping import ColumnMapping

# 1. Chargement des données
try:
    print("--- Mushroom Monitoring Report Generation ---")
    ref_data = pd.read_csv('/data/ref_data.csv')
    prod_data = pd.read_csv('/data/prod_data.csv')

    # Harmonisation des types pour la target (doivent être identiques)
    # Les CSV chargent parfois les booléens comme des strings "True"/"False"
    map_dict = {'True': True, 'False': False, True: True, False: False, 1: True, 0: False}
    ref_data['target'] = ref_data['target'].map(map_dict)
    prod_data['target'] = prod_data['target'].map(map_dict)

    # 2. Configuration du Mapping (Legacy style but used for DataDefinition)
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = ['PCA 1', 'PCA 2', 'PCA 3']

    # 3. Création des objets Dataset (Nouveau style 0.7.20)
    print("Préparation des datasets...")
    ref_dd = create_data_definition(None, ref_data, column_mapping)
    prod_dd = create_data_definition(None, prod_data, column_mapping)
    
    ref_ds = Dataset(ref_data, data_definition=ref_dd)
    prod_ds = Dataset(prod_data, data_definition=prod_dd)

    # 4. Génération du Rapport
    report = Report(metrics=[
        ClassificationPreset(),
        DataDriftPreset(),
    ])

    print("Calcul des métriques (Classification et Drift)...")
    report.run(reference_data=ref_ds, current_data=prod_ds)

    # 5. Sauvegarde dans le Workspace pour l'interface UI
    workspace = Workspace("workspace")
    
    # Récupérer ou créer le projet
    projects = workspace.list_projects()
    project = next((p for p in projects if p.name == "Mushroom Monitoring"), None)
    if project is None:
        project = workspace.create_project("Mushroom Monitoring")
    
    # Utilisation de add_run et as_snapshot comme demandé
    workspace.add_run(project.id, report.as_snapshot())
    print("Rapport généré avec succès dans le workspace.")

except Exception as e:
    print(f"Erreur lors de la génération du rapport : {e}")
    import traceback
    traceback.print_exc()
