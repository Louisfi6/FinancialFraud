import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load
import matplotlib.pyplot as plt

def detecter_fraudes(model_path, dataset_path):
    # Charger le modèle
    model = load(model_path)

    # Charger le nouveau dataset à partir d'un fichier CSV
    new_data = pd.read_csv(dataset_path)

    # Convertir les caractéristiques catégorielles en valeurs numériques
    label_encoder = LabelEncoder()
    new_data['type'] = label_encoder.fit_transform(new_data['type'])

    # Séparation des caractéristiques
    X_new = new_data.drop('isFraud', axis=1)

    # Prédictions sur le nouveau dataset
    y_pred = model.predict(X_new)

    # Convertir les probabilités en 'Y' (fraude) et 'N' (non fraude)
    new_data['FraudeYN'] = ['Y' if x >= 0.5 else 'N' for x in y_pred]

    # Compter le nombre de fraudes dans le nouveau dataset
    nombre_fraudes = new_data['FraudeYN'].value_counts().get('Y', 0)

    return nombre_fraudes

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

# Calculer le nombre de fraudes pour chaque modèle
fraudes_predites = []
for chemin_modele in chemins_des_modeles:
    fraudes_predites.append(detecter_fraudes(chemin_modele, 'clean.csv'))

# Noms des modèles pour l'axe x
model_names = ["Tree", "Lasso", "LightGBM", 
               "Linear", "Logistic", 
               "Gradient", "RandomForest", 
               "Ridge", "XGBoost", "Kmeans",
               "Gaussian"]

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(model_names, fraudes_predites, marker='o', label='Fraudes prédites')
plt.axhline(y=nombre_reel_fraudes, color='r', linestyle='--', label='Fraudes réelles')
plt.xlabel('Modèles')
plt.ylabel('Nbr de fraudes')
plt.title('Accuracy')
plt.legend()
plt.show()
