# INT3405E 56 - FINAL PROJECT 
This project focuses on addressing the [Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) challenge hosted by the Child Mind Institute on Kaggle.

Members: 
1. Vũ Nguyệt Hằng (leader) - 22028079 
2. Ngô Tùng Lâm 	-  22028092

## Repository Structure
```
.
├── experiments
├── feature-eda
```
- `experiments`: contains notebooks related to experimental runs and analyses.
- `feature-eda`: include EDA notebooks for tabular (CSV) and actigraphy  data.

## Our pipeline
### Data processing
- **Tabular data**:
  - We first began by removing negative or abnormally high physical measurements to prevent extreme values from skewing the models.
  - Impute missing values and extract new features based on domain knowledge.
  - Remove low-correlation features.
- **Time series data**: Focused on activity patterns, temporal trends, and daily statistics (mean, median, standard deviation) to create aggregated features.
- **Imputatition**:
  - Categorical features: Missing values are imputed using the most frequent value `SimpleImputer(strategy="most_frequent")`.
  - Numerical features: Using simple imputors such as mean/median imputor `SimpleImputer` or K-Nearest Neighbor imputor `KNNImputer`.
- **Feature transformation**: Apply advanced techniques such as `AutoEncoders`, `Sparse AutoEncoders`, and `PCA` to reduce dimensionality and capture latent representations in the data.
- **Feature selection**: Features were manually and iteratively selected, following the idea of recursive feature elimination.
### Approaches
- **Machine learning models**: We implemented GDBT models (`XGBoost`, `CatBoost`, `LGBM`) from libraries with the same name.
- **Deep Learning models**: We conducted experiments using `TabNet` and `FT-Transformer` to evaluate their performance and suitability for the task.

### Training and evaluation
- **Cross Validation**: `StratifiedKFold` split the data into training and validation sets, ensuring balanced class distribution in each fold.
- **Quadratic Weighted Kappa (QWK)**: Measures the agreement between predicted and actual values, taking into account the ordinal nature of the target variable.
- **Ensemble Learning**: Combine models using 'VotingRegressor' or `StackingRegressor` to improve prediction accuracy and robustness.
- **Early Stopping and Regularization**: Implement early stopping and weight decay for deep learning models to prevent overfitting and improve generalization.
- **Threshold Optimization**: Fine-tune decision thresholds that map continuous predictions to discrete categories (None, Mild, Moderate, Severe).

### Inference
- **Majority Voting**: Combine predictions from multiple models, selecting the majority class to improve reliability and reduce overfitting.

## Experiments and result
| Experiments                                      | Private | Public |
|--------------------------------------------------------|---------|--------|
| Baseline                                         | 0.383   | 0.431  |
| CatBoost - Feature engineering CSV data          | 0.446   | 0.428  |
| CatBoost - Feature engineering full data         | 0.440   | 0.437  |
| FT-Transformer - Feature engineering full data   | 0.334   | 0.343  |
| Stacking Regressor - GBDT - CSV data             | 0.442   | 0.448  |
| Voting Regressor - GBDT - CSV data               | **0.447** | **0.459**  |
| Voting Regressor - GBDT - full data              | 0.431   | 0.431  |
| Major Voting - TabNet + FT-Transformer + GBDT - full data | 0.420 | 0.428  |
| Major Voting - FT-Transformer + GBDT - CSV data  | 0.440   | 0.441  |
| Major Voting - FT-Transformer + GBDT - full data | 0.435   | 0.449  |



