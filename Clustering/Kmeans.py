import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump

#on charge le dataset à partir du fichier CSV
df = pd.read_csv('clean.csv')

#on converti les caractéristiques catégorielles en valeurs numériques
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

#séparation des caractéristiques et de la cible
X = df.drop('isFraud', axis=1)
y = df['isFraud']

#division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#utiliser K-Means pour effectuer le clustering
kmeans = KMeans(n_clusters=2, random_state=42)  #on choisi le nombre de clusters approprié
y_clusters = kmeans.fit_predict(X_train)

#assigner les clusters aux classes majoritaires (0 ou 1)
cluster_labels = [0 if sum(y_clusters == cluster) > len(y_clusters) / 2 else 1 for cluster in set(y_clusters)]

#assigner les clusters aux données
y_train_pred = [cluster_labels[cluster] for cluster in y_clusters]

#évaluation du modèle
precision = accuracy_score(y_train, y_train_pred)

#imprimer le rapport de classification et la précision
print("Rapport de classification :\n", classification_report(y_train, y_train_pred))
print("Précision :", precision)

# Enregistrer le modèle K-Means
dump(kmeans, 'Modele_KMeans.joblib')
