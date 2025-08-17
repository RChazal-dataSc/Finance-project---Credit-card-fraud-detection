💳 Credit Card Fraud Detection

🎯 Objectif

Détecter automatiquement les transactions bancaires frauduleuses parmi un grand volume de transactions.
Dataset utilisé : Kaggle – Credit Card Fraud Detection.

Transactions totales : 284 807

Fraudes : 492 (0.17%)

Particularité : classes extrêmement déséquilibrées

🔧 Pipeline
1. Exploration et préparation des données

Vérification des valeurs manquantes et infinies → aucune.
Analyse de la distribution des variables et de la corrélation avec la cible.
Normalisation des données avec StandardScaler.
Suppression de la variable Time (faible intérêt).

2. Gestion du déséquilibre

Application de Borderline-SMOTE pour rééquilibrer les classes (492 → 284k échantillons synthétiques).

3. Modélisation

Trois approches principales ont été testées :

Isolation Forest (non supervisé, adapté aux anomalies rares) → résultats insatisfaisants.

Modèles supervisés classiques :

Régression Logistique

Random Forest

Réseau de Neurones (ANN) :
Input layer (64 neurones, ReLU)
Hidden layers (32 & 16 neurones, ReLU + Dropout)
Output layer (sigmoïde)
Optimiseur Adam

📊 Résultats
Modèle	Accuracy	Precision	Recall	F1-score
Isolation Forest	❌	-	-	-
Régression Logistique	0.986	0.985	0.986	0.986
Random Forest	0.9998	0.9998	0.9998	0.9998
ANN (Keras)	0.9988	0.9984	0.9991	0.9988

👉 Meilleures performances : Random Forest & ANN

La Random Forest reste la plus simple et robuste.
L’ANN a produit des résultats comparables, tout en montrant un bon équilibre précision/rappel.

🚀 Déploiement (bonus)

Un modèle Random Forest a été sauvegardé (pickle) et intégré dans une API Flask minimale :
Entrée : caractéristiques d’une transaction
Sortie : prédiction binaire

0 → Transaction normale
1 → Fraude détectée

📚 Bibliothèques utilisées

pandas, numpy, matplotlib, seaborn
scikit-learn, imbalanced-learn
tensorflow.keras
flask
etc...

🧠 Perspectives

Amélioration du réseau de neurones (plus d’epochs, tuning des hyperparamètres).
Test d’algorithmes de boosting (XGBoost, LightGBM).
Déploiement d’une API complète avec documentation (Swagger / FastAPI).
