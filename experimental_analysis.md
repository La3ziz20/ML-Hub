# Experimental Analysis: ML Model Benchmarking

## Which model is best?
Based on the direct quantitative benchmark of the `cleaned_dataset.csv` targeting the `Category` column, **Random Forest** dramatically outperformed all other algorithms. 
In our experiment tracking, Random Forest achieved an overwhelming **87.1% Accuracy**, with exceptionally balanced Precision (86.9%) and Recall (87.1%) and F1 Score (86.7%).

## Why?
Random Forest relies on an ensemble approach of decision trees. This allows it to natively detect and prioritize non-linear interactions within the dataset's features (such as subtle relationships between `Mileage`, `Engine_volume`, `Prod._year`, and `Price`) far more effectively than linear models. 

Conversely, **Linear/Logistic Regression** heavily struggled (yielding a poor **46.5% Accuracy** and 21.6% precision). This is because it attempts to fit a straight decision hyperplane to a highly dimensional, inherently non-linear mapping space consisting of heavily skewed automotive variables. **KNN** fared slightly better (**53.8% Accuracy**) but suffered because the distance between normalized auto-parameters in a multi-dimensional space becomes indistinguishable (often referred to as the *curse of dimensionality*), thereby muddying its nearest-neighbor predictions.

## Which hyperparameters matter most?
1. **Random Forest:** The number of estimators (`n_estimators`) dictates the variance reduction. More trees typically yield better generalization, plateauing eventually.
2. **KNN:** `n_neighbors` is the most significant constraint needed to curb overfitting (if set too low) or underfitting (if set too high).
3. **SVM:** Both the `C` penalty and the `Kernel` (RBF vs Linear) warp the geometric classification boundaries. Tuning `C` accurately penalizes the structural risk to widen or tighten the training margins.

## Did PCA improve performance? Compare: With PCA / Without PCA
- **Without PCA:** The Random Forest algorithm achieved **87.1% Accuracy**. The raw dimensional space preserved subtle non-linear thresholds between complex features like `Mileage` and `Engine Volume` that are deeply tied to specific automobile frames/categories.
- **With PCA (2 Components):** Model performance drops dramatically (down to roughly **46% Accuracy** on cross-validation testing). 

This demonstrates that while the highest variance principal components successfully collapse the dataset visually, they inherently sacrifice lower-variance features that contained highly discriminative classification patterns needed to separate the overlapping clusters of Sedan and Jeep. In conclusion, PCA did **not** improve predictive performance; it merely traded heuristic depth for computational and visual simplicity.
