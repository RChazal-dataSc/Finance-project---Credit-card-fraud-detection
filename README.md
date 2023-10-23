# Finance-project---Credit-card-fraud-detection
ML and DL script for Fraudulous credit card transaction detection

- Quel était le problème ?

Parmi un jeu de données de transactions bancaires, identifier les fraudes.

- Quelle méthode avez-vous suivi pour le résoudre ? Quels outils avez-vous choisi ?

Analyse et nettoyage du jeu de données, matrice de correlation, oversampling SMOTE, standard scaler, encodage, isolation forest, random forest, ANN tensorfow.keras.
Pour le fun, création d'une API très simple avec Flask.

- Éventuellement, quelques remarques sur le code programmé.

Pas grand chose à dire, ce code n'a rien de particulier

- Des exemples de résultats obtenus.

L'isolation forest a été décevante, malgré le fait que cet algo est adapté pour la détection de cibles très minoritaires. La random forest a été l'algo le plus performant (acc_train :  0.986 acc_test : 0.9859592847395152, F1score_train :  0.986 F1score_test :  0.9859590684070437), avec très peu de mal classés.
Le réseau de neurones n'a pas fait mieux, je l'ai donc écarté.
