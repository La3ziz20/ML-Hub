# Rapport Tâche 4 : Classification avec Random Forest

## 1. Explorer l'importance des features
* **Graphique** : L'image `feature_importance.png` a été générée et sauvegardée dans le dossier `backend` et loggée via MLflow.
* **Top 3 variables** : 
  1. **Price** (0.0506)
  2. **ID** (0.0472)
  3. **Prod. year** (0.0314)
* **Interprétation** : L'importance du prix et de l'année de production est logique car ces variables définissent souvent le type de véhicule (ex: les véhicules haut de gamme ou récents appartiennent à certaines catégories spécifiques). Cependant, la présence de la colonne `ID` en deuxième position est **anormale et indique un problème conceptuel**. L'`ID` est un identifiant unique qui n'a aucun sens pour la classification. Sa présence pousse le modèle à faire du *par cœur* (overfitting en mémorisant les identifiants). Il sera crucial de supprimer cette colonne avant de ré-entraîner les modèles en production.

## 2. Stabilité des prédictions
* **Accuracies (5 random_state différents)** : `[0.8692, 0.8653, 0.8672, 0.8648, 0.8679]`
* **Moyenne** : 86.69% | **Écart-type** : 0.0016
* **Conclusion** : L'écart-type est extrêmement faible (0.16%). Cela montre qu'en modifiant la graine aléatoire, la performance reste presque identique. Le modèle Random Forest est donc très **robuste** et très stable face aux variations d'échantillonnage de l'algorithme.

## 3. Analyse des erreurs
* **Total d'erreurs observées** : 501 erreurs sur 3848 exemples de test.
* **Exemples concrets** :
  * Erreur 1 : Réel = `Hatchback`, Prédit = `Sedan` (Prix: 12858, Année: 2011)
  * Erreur 2 : Réel = `Sedan`, Prédit = `Jeep` (Prix: 18817, Année: 2008)
  * Erreur 3 : Réel = `Sedan`, Prédit = `Hatchback` (Prix: 18817, Année: 2013)
* **Patterns observés** : Le modèle a principalement du mal à différencier les véhicules `Sedan` (Berlines) avec les `Hatchback` (Compactes) ou les `Jeep` (SUV). C'est cohérent, car ces catégories sont souvent très proches au niveau des caractéristiques techniques comme le volume du moteur, le prix ou l'année de production. Les frontières géométriques entre ces classes sont très floues.

## 4. Biais et Variance

| n_estimators | max_depth | Train Accuracy | Test Accuracy | Biais  | Variance |
|--------------|-----------|----------------|---------------|--------|----------|
| 10           | 5         | 0.4989         | 0.4914        | 0.5011 | 0.0074   |
| 10           | 10        | 0.5859         | 0.5772        | 0.4141 | 0.0087   |
| 10           | None      | 0.9932         | 0.8547        | 0.0068 | 0.1385   |
| 50           | 5         | 0.4555         | 0.4493        | 0.5445 | 0.0061   |
| 50           | 10        | 0.5778         | 0.5702        | 0.4222 | 0.0076   |
| 50           | None      | 0.9999         | 0.8693        | 0.0001 | 0.1307   |
| 100          | 5         | 0.4553         | 0.4493        | 0.5447 | 0.0060   |
| 100          | 10        | 0.5767         | 0.5699        | 0.4233 | 0.0068   |
| 100          | None      | 0.9999         | 0.8698        | 0.0001 | 0.1301   |

* **Quel paramétrage montre overfitting ?** 
  L'utilisation de `max_depth=None` montre clairement un surapprentissage. Le modèle atteint 99.99% d'accuracy en entraînement, mais plafonne à 86.9% en test. Le biais est quasi nul (0.0001), mais la **variance est la plus élevée (~0.13)**.
* **Quel paramétrage montre underfitting ?** 
  L'utilisation de `max_depth=5` (arbre trop peu profond) cause un sous-apprentissage sévère. La précision est mauvaise autant en entraînement (45%) qu'en test (44%). Le **biais est très élevé (~0.54)**.
* **Quel paramétrage semble équilibré ?** 
  Si l'on veut maximiser le score, `max_depth=None` avec `100` estimateurs donne les meilleures performances en test. Cependant, pour un modèle équilibré au sens académique (minimiser la différence Train/Test tout en gardant un bon score), un paramétrage intermédiaire (ex: `max_depth=15` ou `20`) aurait été optimal. Parmi nos paramètres, `max_depth=None` avec `n_estimators=100` compense le plus sa variance grâce à la quantité d'arbres.

## 5. Comparaison avec l'algorithme Arbre de décision
* **Decision Tree** : Train Accuracy = 99.99% | Test Accuracy = **84.54%**
* **Random Forest** (100 est., depth=None) : Train Accuracy = 99.99% | Test Accuracy = **86.98%**
* **Conclusion** : Le Random Forest surpasse nettement le simple Arbre de décision. L'Arbre de décision seul souffre d'une variance encore plus forte car il construit une structure très complexe collée aux données d'entraînement. Le Random Forest, grâce au mécanisme de *bagging* (moyenne de 100 arbres entraînés sur des sous-échantillons aléatoires), permet de lisser cette variance et gagne ainsi **+2.4%** de précision sur le jeu de données invisible (Test Set).

*(Toutes les expériences présentées ont été loggées et sauvegardées dans le répertoire MLflow du projet).*
