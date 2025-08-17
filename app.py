#installer API
#Librairies
from sklearn.metrics import precision_score, f1_score
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import json


# Charger le modèle Isolation Forest pré-entrainé à partir du fichier pickle
with open('creditcard_fraud_rf.sav', 'rb') as model_file:
    model = pickle.load(model_file)

    
#API
app = Flask(__name__)
    
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lire le fichier CSV depuis la requête
        file = request.files['file']
        if not file:
            return jsonify({"error": "Aucun fichier n'a été envoyé"})

        # Lire les données CSV en un DataFrame
        df = pd.read_csv(file)

        # Supprimer les colonnes inutiles si elles existent
        if "Time" in df.columns:
            df = df.drop("Time", axis=1)
        if "Class" in df.columns:
            y_true = df["Class"].values  # garder de côté pour évaluation
            df = df.drop("Class", axis=1)
        else:
            y_true = None


        # Prétraitement des données (vous devrez adapter ceci en fonction de votre modèle)
        sc = StandardScaler()
        df = sc.fit_transform(df)

        # Effectuer les prédictions
        predictions = model.predict(df)
        
        
        # Conversion en labels texte
        labels = ["transaction normale" if pred == 0 else "fraude détectée" for pred in predictions]

        # Résultats ligne par ligne
        results = []
        for i in range(len(predictions)):
            results.append({
                "prediction": int(predictions[i]),
                "label": labels[i],
                "true_label": int(y_true[i]),
                "correct": bool(predictions[i] == y_true[i])
            })


        # Métriques globales
        precision = precision_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        global_metrics = {
            "precision": float(precision),
            "f1_score": float(f1)
        }


        # JSON final
        output = {
            "results": results,
            "global_metrics": global_metrics
        }


        # Créer un DataFrame avec les prédictions et les indicateurs de performance
        #result_df = pd.DataFrame({
        #    "predictions": list(predictions),
        #    "labels": y_pred_labels[0],
        #    "precision": [precision] * len(predictions),
        #    "f1_score": [f1] * len(predictions)
        #})

        # Créez un dictionnaire pour stocker les résultats
        #results = {
        #    "predictions": list(predictions),
        #    "labels": list(y_pred_labels[0]),
        #    "precision": float(precision),  # Convertissez la précision en nombre à virgule flottante
        #    "f1_score": float(f1)  # Convertissez le F1-score en nombre à virgule flottante
        #}

        # Enregistrez les résultats dans un fichier JSON dans le répertoire de travail
        nom_fichier_json = "donnees.json"

        # Enregistrez les données dans le fichier JSON
        with open(nom_fichier_json, "w") as fichier_json:
            json.dump(results, fichier_json)

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)






