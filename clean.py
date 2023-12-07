import pandas as pd

def nettoyer_csv(chemin_entree, chemin_sortie):
    # Charger le CSV dans un DataFrame pandas
    df = pd.read_csv(chemin_entree)

    # Filtrer les lignes où oldbalance ou newbalance n'est pas égal à 0
    columns_to_drop = ['nameOrig', 'nameDest']
    df = df.drop(columns_to_drop, axis=1)
    # Enregistrer le DataFrame nettoyé dans un nouveau fichier CSV
    df.to_csv(chemin_sortie, index=False)

if __name__ == "__main__":
    chemin_entree = "PS_20174392719_1491204439457_log.csv"
    chemin_sortie = "clean.csv"

    nettoyer_csv(chemin_entree, chemin_sortie)