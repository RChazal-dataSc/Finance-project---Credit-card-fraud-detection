#installer API
#Librairies
from sklearn.metrics import precision_score, f1_score
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import json


# Charger le modèle Isolation Forest pré-entrainé à partir du fichier pickle
with open('finance-project_credit-card-fraud-detection.sav', 'rb') as model_file:
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

        # Prétraitement des données (vous devrez adapter ceci en fonction de votre modèle)
        sc = StandardScaler()
        df = sc.fit_transform(df)

        # Effectuer les prédictions
        predictions = model.predict(df)
        
        
        # Mapper les prédictions (0 et 1) aux libellés ("transaction normale" et "fraude détectée")
        y_pred_labels = ["transaction normale" if pred == 0 else "fraude détectée" for pred in predictions]
        y_pred_labels = pd.DataFrame(y_pred_labels)
        

        # Créer un DataFrame avec les prédictions et les indicateurs de performance
        result_df = pd.DataFrame({
            "predictions": list(predictions),
            "labels": y_pred_labels[0],
            "precision": [precision] * len(predictions),
            "f1_score": [f1] * len(predictions)
        })

        # Créez un dictionnaire pour stocker les résultats
        results = {
            "predictions": list(predictions),
            "labels": list(y_pred_labels[0]),
            "precision": float(precision),  # Convertissez la précision en nombre à virgule flottante
            "f1_score": float(f1)  # Convertissez le F1-score en nombre à virgule flottante
        }

        # Enregistrez les résultats dans un fichier JSON dans le répertoire de travail
        nom_fichier_json = "donnees.json"

        # Enregistrez les données dans le fichier JSON
        with open(nom_fichier_json, "w") as fichier_json:
            json.dump(results, fichier_json)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)






