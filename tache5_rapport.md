# Compte Rendu Tâche 5 : Pratiques MLOps

Ce document répond aux questions de réflexion posées lors de la mise en place du pipeline MLOps pour le projet Machine Learning.

## Partie 1 — Tracking des Expérimentations

**Q1. Quelle est la différence entre `mlflow.log_param()` et `mlflow.log_metric()` ?**
* `log_param()` sert à enregistrer les hyperparamètres (ex: `max_depth`, `n_estimators`, `learning_rate`) ou toute configuration statique du modèle qui ne change pas pendant l'entraînement. 
* `log_metric()` sert à enregistrer les valeurs dynamiques de performance issues de l'évaluation (ex: `accuracy`, `f1_score`, `loss`), qui mesurent la qualité de l'algorithme et peuvent évoluer.

**Q2. Pourquoi est-il important de nommer ses runs (`run_name`) ?**
Nommer les runs permet de filtrer, chercher et comparer facilement les expérimentations dans l'interface MLflow (UI). Un nom comme `rf_depth10_est100` est beaucoup plus descriptif et exploitable qu'un UUID généré aléatoirement.

**Q3. Que se passe-t-il si vous exécutez deux fois le même script sans changer le `run_name` ?**
MLflow créera deux runs distincts (avec deux `run_id` différents) qui porteront le même `run_name`. Cela peut créer de la confusion dans l'UI lors de l'analyse, d'où l'importance de versionner ou d'horodater les noms si on relance exactement le même script.

## Partie 2 — Comparaison d'Expérimentations

**Q4. Quel modèle obtient le meilleur compromis accuracy / f1_score ? Justifiez.**
Suite à nos entraînements (`train_batch.py`), le modèle **RandomForest avec `n_estimators=100` et `max_depth=None`** offre souvent le meilleur compromis. Bien que le Gradient Boosting soit compétitif, le Random Forest gère très bien les classes déséquilibrées et maintient un F1-score pondéré solide tout en maximisant l'accuracy (~86.9%). 

**Q5. Le graphique Parallel Coordinates révèle-t-il une corrélation entre `max_depth` et accuracy ?**
Oui, on observe clairement sur le graphique Parallel Coordinates que les faibles valeurs de `max_depth` (ex: 3 ou 5) aboutissent aux lignes les plus basses en termes d'accuracy (underfitting sévère). À l'inverse, lorsque `max_depth` s'approche de 15 ou de `None`, les lignes pointent vers les plus hauts scores d'accuracy.

**Q6. Comment MLflow permet-il la reproductibilité par rapport à un simple `print()` des métriques ?**
Un `print()` disparaît une fois le terminal fermé. MLflow stocke de façon permanente :
1. Le code exact utilisé (via Git commit hash si intégré)
2. Les paramètres d'entrée
3. Les données de sortie et les métriques
4. L'artefact binaire du modèle (`.joblib` ou format `sklearn`), garantissant qu'on peut recharger exactement ce même modèle un an plus tard.

## Partie 3 — Model Registry

**Q7. Pourquoi séparer les étapes Staging et Production dans un registre de modèles ?**
`Staging` est un environnement de pré-production. On y déploie le modèle pour que l'équipe d'intégration (QA) ou des scripts de tests A/B s'assurent que l'API répond correctement et que la latence est bonne. Une fois validé, on le passe en `Production` pour qu'il reçoive le trafic réel des clients. Cela évite d'exposer un modèle défectueux au public.

**Q8. Que se passe-t-il si l'on archive une version en Production ? Quel impact opérationnel ?**
Si on archive la version actuellement en `Production` sans la remplacer, l'API de prédiction risque de tomber ou de retourner des erreurs (404), car l'alias "Production" ne pointera plus vers un modèle actif. Le service sera interrompu.

**Q9. Comment le Registry facilite-t-il le rollback vers une version précédente ?**
Toutes les versions sont numérotées (v1, v2, v3...). Si la v3 (Production) se dégrade, un simple clic dans l'UI ou une commande CLI (`client.transition_model_version_stage`) permet de repasser la v2 de `Archived` à `Production` instantanément.

## Partie 4 — Serving et API REST

**Q10. Quel est l'avantage d'un serving MLflow natif vs FastAPI personnalisé ?**
* **MLflow natif** : Déploiement en 1 ligne de commande. Prêt à l'emploi. Idéal pour des tests rapides ou des environnements standardisés.
* **FastAPI** : Offre une personnalisation totale. On peut ajouter de la sécurité (JWT), du pré/post-traitement complexe, et une documentation Swagger native.

**Q11. Comment géreriez-vous le rechargement automatique d'un nouveau modèle en Production ?**
Je mettrais en place un webhook (ou une tâche cron) qui écoute les changements d'état dans le Model Registry MLflow. Dès qu'une nouvelle version passe en `Production`, le script télécharge le nouvel artefact et redémarre le conteneur Docker FastAPI ou utilise une variable globale en mémoire pour recharger le `.predict()` dynamiquement.

**Q12. Quels headers HTTP ajouteriez-vous pour sécuriser l'endpoint en production réelle ?**
* `Authorization: Bearer <token>` pour l'authentification (JWT, OAuth2).
* `X-API-Key` pour limiter le rate-limiting par client.
* `Strict-Transport-Security` pour forcer le HTTPS.

## Partie 6 — Détection du Data Drift

**Q13. Quelle est la différence entre data drift et concept drift ? Donnez un exemple concret avec vos propres données.**
* **Data Drift** (Covariate Shift) : Les variables d'entrée changent de distribution, mais la règle pour deviner la sortie reste la même. *Exemple* : On avait des voitures d'occasion vendues majoritairement avec 4 cylindres en 2015. Aujourd'hui on reçoit des données de voitures électriques sans cylindre, ou de gros SUV avec 8 cylindres. 
* **Concept Drift** : La définition même de la cible change. *Exemple* : Une voiture vendue 20 000$ en 2010 était classée "Haut de gamme". Avec l'inflation, en 2024, une voiture à 20 000$ est classée "Économique". La relation Prix -> Catégorie a changé.

**Q14. Le KS-test et Evidently identifient-ils les mêmes features comme driftées ? Pourquoi ?**
Généralement oui, car Evidently utilise le test de Kolmogorov-Smirnov (KS) en interne pour les variables numériques continues par défaut. Le KS-test compare les distributions cumulées empiriques. Les résultats p-value seront donc quasi-identiques.

**Q15. Quel seuil de drift choisiriez-vous pour votre projet ? Justifiez selon le domaine métier.**
Le prix et l'année de production (features importantes) sont très volatiles. Je choisirais un seuil autour de **25% à 30%** de `drift_share`. Si l'on est trop sensible (ex: 5%), on risque de réentraîner le modèle tous les jours à cause des légères variations naturelles des prix du marché de l'automobile.

**Q16. Sans pipeline MLOps automatisé, comment détecteriez-vous ce drift en pratique ? Quels sont les risques pour une application en production ?**
Sans MLOps, on le détecterait bien plus tard, via des plaintes des utilisateurs, ou une chute de la rentabilité (chiffre d'affaires). Le risque principal est la **dégradation silencieuse** : le modèle continue de retourner des prédictions formatées correctement, le code ne crashe pas (pas de bugs IT), mais les décisions business générées sont fausses, entraînant des pertes financières.

---
## Pipeline Fermé (Schéma)
```text
 Données brutes
       |
 [Prétraitement + DVC] ───> versioning données & code
       |
 [Entraînement + MLflow Tracking] ───> params, métriques, artefacts
       |
 [MLflow Model Registry] ───> Staging -> Production
       |
 [Serving API REST] ───> mlflow models serve / FastAPI
       |
 [Evidently + KS-test + MLflow] ───> drift_share, rapport HTML
       |
 drift > seuil (30%) ? ── OUI ──> retour Entraînement (boucle fermée)
                     ── NON ──> surveillance continue
```
