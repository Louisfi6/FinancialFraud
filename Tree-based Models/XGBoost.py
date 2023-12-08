import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Import XGBoost Regressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump

#on charge le dataset à partir du fichier CSV
df = pd.read_csv('clean.csv')

#on convertir les caractéristiques catégorielles en valeurs numériques
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

#séparation des caractéristiques et de la cible
X = df.drop('isFraud', axis=1)
y = df['isFraud']

#division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#création du modèle XGBoost Regressor
model = XGBRegressor(n_estimators=100, random_state=42)

#entraînement du modèle
model.fit(X_train, y_train)

#prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

#évaluation du modèle
mse = mean_squared_error(y_test, y_pred)

#imprimer l'erreur quadratique moyenne (précision)
print("Erreur quadratique moyenne :", mse)

#enregistrer le modèle
dump(model, 'Modele_XGBoostRegressor.joblib')
