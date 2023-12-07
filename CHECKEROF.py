import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load
import time
from sklearn.metrics import accuracy_score

def detecter_fraudes(model_path, dataset_path):
    # Charger le modèle
    start_time = time.time()
    model = load(model_path)
    load_time = time.time() - start_time

    # Charger le nouveau dataset à partir d'un fichier CSV
    new_data = pd.read_csv(dataset_path)

    # Convertir les caractéristiques catégorielles en valeurs numériques
    label_encoder = LabelEncoder()
    new_data['type'] = label_encoder.fit_transform(new_data['type'])

    # Séparation des caractéristiques
    X_new = new_data.drop('isFraud', axis=1)

    # Prédictions sur le nouveau dataset
    start_time = time.time()
    y_pred = model.predict(X_new)
    predict_time = time.time() - start_time

    # Convertir les probabilités en 'Y' (fraude) et 'N' (non fraude)
    new_data['FraudeYN'] = ['Y' if x >= 0.5 else 'N' for x in y_pred]

    # Convertir les prédictions continues en format binaire
    y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

    # Calculer la précision du modèle
    accuracy = accuracy_score(new_data['isFraud'], y_pred_binary)

    # Compter le nombre de fraudes dans le nouveau dataset
    nombre_fraudes = new_data['FraudeYN'].value_counts().get('Y', 0)

    return load_time, predict_time, accuracy, nombre_fraudes

# Charger le dataset à partir d'un fichier CSV
df = pd.read_csv('clean.csv')

# Compter le nombre de 1 dans la colonne isFraud
nombre_reel_fraudes = df['isFraud'].sum()

# Liste des chemins vers les fichiers .joblib
chemins_des_modeles = [
    'Joblib/Modele_DecisionTree.joblib',
    'Joblib/Modele_LassoRegression.joblib',
    'Joblib/Modele_LightGBMRegressor.joblib',
    'Joblib/Modele_LinearRegression.joblib',
    'Joblib/Modele_LogisticRegression.joblib',
    'Joblib/Modele_Optimise_GradientBoostingRegressor.joblib',
    'Joblib/Modele_RandomForest.joblib',
    'Joblib/Modele_RidgeRegression.joblib',
    'Joblib/Modele_XGBoostRegressor.joblib',
    'Joblib/Modele_KMeans.joblib',
    'Joblib/Modele_GaussianMixture.joblib'

]

# Calculer le temps de chargement, le temps de prédiction, la précision et le nombre de fraudes pour chaque modèle
resultats = []
for chemin_modele in chemins_des_modeles:
    load_time, predict_time, accuracy, fraudes = detecter_fraudes(chemin_modele, 'clean.csv')
    resultats.append((load_time, predict_time, accuracy, fraudes))

# Afficher les résultats
for i, chemin_modele in enumerate(chemins_des_modeles):
    model_name = chemin_modele.split('/')[-1].split('.')[0]
    load_time, predict_time, accuracy, fraudes = resultats[i]
    print(f"{model_name} model: {load_time:.2f}s (Chargement), {predict_time:.2f}s (Analyse), {accuracy:.6f} Precision, {fraudes} fraudes predites")
