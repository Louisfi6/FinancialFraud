import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  # Import Gradient Boosting Regressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump

#charger le dataset à partir du fichier CSV
df = pd.read_csv('clean.csv')

#échantillon d'une fraction des données
df_sample = df.sample(frac=0.1, random_state=42)

#on converti les caractéristiques catégorielles en valeurs numériques
label_encoder = LabelEncoder()
df_sample['type'] = label_encoder.fit_transform(df_sample['type'])

#séparation des caractéristiques et de la cible
X = df_sample.drop('isFraud', axis=1)
y = df_sample['isFraud']

#division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#création du modèle Gradient Boosting Regressor avec des paramètres optimisés
model = GradientBoostingRegressor(n_estimators=50, max_depth=5, subsample=0.8, random_state=42)

#entraînement du modèle
model.fit(X_train, y_train)

#prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

#évaluation du modèle
mse = mean_squared_error(y_test, y_pred)

#imprimer l'erreur quadratique moyenne
print("Erreur quadratique moyenne :", mse)

#enregistrer le modèle
dump(model, 'Modele_Optimise_GradientBoostingRegressor.joblib')
