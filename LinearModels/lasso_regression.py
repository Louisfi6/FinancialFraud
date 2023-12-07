import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso  # Import Lasso Regression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Charger le dataset à partir d'un fichier CSV
df = pd.read_csv('clean.csv')

# Convertir les caractéristiques catégorielles en valeurs numériques
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Séparation des caractéristiques et de la cible
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Création du modèle de régression Lasso
model = Lasso()

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Convertir les probabilités en 0 (non fraude) et 1 (fraude)
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

# Évaluation du modèle
precision = accuracy_score(y_test, y_pred_binary)

# Imprimer le rapport de classification et la précision
print("Rapport de classification :\n", classification_report(y_test, y_pred_binary))
print("Précision :", precision)

# Enregistrer le modèle
dump(model, 'Modele_LassoRegression.joblib')
