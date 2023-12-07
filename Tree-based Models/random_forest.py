import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Charger le dataset à partir du clean.csv
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

# Création du modèle Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
precision = accuracy_score(y_test, y_pred)

# Imprimer le rapport de classification et la précision
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("Précision :", precision)

# Enregistrer le modèle
dump(model, 'Modele_RandomForest.joblib')
