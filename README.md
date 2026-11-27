# Predicting Diabetes Risk: Applying Machine Learning to Preventative Health

## 1. Project Overview

**Business context.** A lifestyle and wellness app wants to screen new users for diabetes risk during onboarding. Many people with prediabetes don't know they have it, and long health questionnaires make users drop out. The app needs a **short risk check** that still finds most people at risk.

**Business question:** *What can be done to reduce your chances of getting diabetes?*

**Objectives**
- Identify which health indicators are most associated with (pre)diabetes.
- Build a classification model that flags users at risk, so the app can suggest a blood sugar test and prevention tips.
- Keep the questionnaire as short as possible without losing too much screening power.

**Machine Learning problem:** binary classification (no diabetes vs. prediabetes/diabetes).

**Final Presentation:** [Presentation](https://canva.link/5pqxi0zxyezk1ce)

## 2. Research Questions

The 22 features were split into 4 themes. Each team member explored one group and tested one hypothesis:

| # | Feature group | Research question |
|---|---|---|
| H1 | Demographics & Socioeconomic | Do age, sex, education and income predict (pre)diabetes? |
| H2 | Clinical Conditions | Are clinical indicators (blood pressure, cholesterol, BMI, heart disease, stroke) the strongest predictors overall? |
| H3 | Lifestyle & Habits | Among people with a high BMI (≥ 30), do those with healthy habits have a lower (pre)diabetes rate? |
| H4 | Self-Reported Health | Does self-rated general health differ between people with and without diabetes? |

**Target variable:** `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes). Prediabetes is included because the app's goal is early prevention, and these are the users who benefit most from changing their habits.

**What counts as a useful model:** high **recall**. For a screening tool, missing a person at risk costs more than sending someone for an extra test that turns out negative.

## 3. Dataset Description

- **Source:** [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (UCI / Kaggle), based on the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 telephone survey.
- **File used:** `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`
- **Size:** 253,680 respondents × 22 columns (21 features + target).
- **Class balance:** 86.1% no diabetes, 13.9% (pre)diabetes.

| Group | Features |
|---|---|
| Demographics & Socioeconomic | `Age` (13 age bands), `Sex`, `Education` (1–6), `Income` (1–8) |
| Clinical Conditions | `HighBP`, `HighChol`, `CholCheck`, `BMI`, `Stroke`, `HeartDiseaseorAttack` |
| Lifestyle & Habits | `PhysActivity`, `Fruits`, `Veggies`, `Smoker`, `HvyAlcoholConsump` |
| Self-Reported Health | `GenHlth` (1–5), `MentHlth` (0–30 days), `PhysHlth` (0–30 days), `DiffWalk` |
| Healthcare access | `AnyHealthcare`, `NoDocbcCost` |

**Limitations**
- All answers are self-reported, so they may be inaccurate (e.g. weight, alcohol use).
- The data is cross-sectional (one point in time): we can find associations, not causes.
- Habit questions are binary. For example, `Fruits` = 1 only means eating fruit at least once a day, not eating healthily overall.

### Main Data Quality Issues

- **Missing values:** none.
- **Data types:** values were stored as floats and converted to integers.
- **Duplicates:** 24,206 duplicated rows (9.5%). With 22 mostly binary/ordinal columns, different people can give identical answers, so these are not data errors.
- **Outliers:** `BMI` ranges from 12 to 98. The IQR rule flags 9,820 rows (3.9%) above 41.5, and 805 rows are above 60.
- **Class imbalance:** only 13.9% of respondents are in the positive class.

### Data Cleaning and Preparation

- **Duplicates kept:** removing them would remove real respondents with common answer profiles.
- **BMI outliers kept:** a BMI above 40 means severe obesity, which is real and is exactly the highest-risk group. Only extremely rare values (e.g. 98) may be reporting errors.
- **Encoding:** no encoding needed; all features are already numeric (binary or ordinal).
- **Feature selection:** each member started from their own group. `CholCheck` was flagged as a candidate to drop (r = 0.06; it mostly reflects healthcare access). In the demographics analysis, a combined socioeconomic score correlated 0.94 with `Income`, so only the score was kept.
- **Class imbalance:** handled with `class_weight="balanced"` (or `sample_weight` for Gradient Boosting) instead of resampling.
- **Data leakage prevention:** the data was split into train and test sets **before** any scaling or model fitting. Scaling was done inside scikit-learn `Pipeline`s, so it is fitted on the training data only. Class weights are computed from the training labels only.

## 4. Methodology

**Exploratory Data Analysis.** For each group we compared the (pre)diabetes rate across feature values (`groupby`, `crosstab`), looked at correlations with the target and plotted distributions. For H3, we repeated the comparison inside the BMI ≥ 30 group to see whether habits still matter when weight is held at a similar level.

**Hypothesis tests**
- H3: chi-square test of independence for each habit within BMI ≥ 30, with **Cramér's V** as effect size (α = 0.05). With large samples almost everything is "significant", so the effect size is what tells us whether a difference matters.
- H4: Mann-Whitney U test for `GenHlth` between people with and without diabetes (ordinal variable, not normally distributed).

**Data splitting.** 80/20 train/test split, **stratified** on the target so both sets keep the 13.9% positive rate, with `random_state=42` for reproducibility. All members used the same split to make results comparable.

**Preprocessing.** `StandardScaler` for distance- and coefficient-based models (KNN, Logistic Regression). Tree-based models do not need scaling.

**Models compared**
- Baseline: `DummyClassifier` (always predicts "no diabetes")
- K-Nearest Neighbors
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

These cover a simple linear model (easy to interpret), distance-based learning and tree ensembles (which capture non-linear effects and interactions).

**Evaluation metrics.** The main metric is **recall** (share of real (pre)diabetes cases found), with precision, F1, ROC-AUC and confusion matrices as support. Accuracy is misleading here: a model that predicts "no diabetes" for everyone is 86% accurate but finds nobody.

**Model validation.** Train and test recall were compared to check for overfitting. For the clinical Random Forest, hyperparameters were tuned with 3-fold `RandomizedSearchCV` (ROC-AUC) on the training data.

**Decision threshold.** For the lifestyle model, a stricter threshold (0.6) was compared with the default (0.5). It cut false positives by 41% but dropped recall from 0.66 to 0.47, so 0.5 was kept. For the self-reported model, the threshold was lowered to 0.2 to reduce missed cases.

## 5. Key Findings

### H1: Demographics & Socioeconomic
- (Pre)diabetes is more common in older, lower-income and less-educated people; sex barely matters.
- All correlations with the target are weak (< 0.2).
- On its own this group is a poor predictor: Logistic Regression found only 28 of 7,069 cases (recall 0.4%).

### H2: Clinical Conditions *(confirmed)*
- `HighBP`: 24.5% (pre)diabetes rate vs 6.0% without high blood pressure (> 4× risk).
- `HighChol`: 22.0% vs 8.0% (~2.8× risk).
- BMI ≥ 30 (obese): 23.4% vs 5.7% for normal weight.
- About 1 in 3 people with heart disease (33.0%) or a stroke (31.8%) has (pre)diabetes.
- Tuned Random Forest on all features: **recall 77.4%**, precision 31.0%, test ROC-AUC 0.824 (CV 0.828, so no overfitting).
- Three clinical features are in the overall top 5 by importance: `HighBP` (#2), `BMI` (#3), `HighChol` (#5).
- A model with only the 6 clinical features keeps most of the power: recall 74.1%, ROC-AUC 0.780.

### H3: Lifestyle & Habits *(supported for physical activity)*
- Inactive people have almost twice the (pre)diabetes rate of active people (21.1% vs 11.6%).
- Among people with BMI ≥ 30, the gap stays the same: 29.9% vs 20.3% (9.6 points).
- The fruit and vegetable gaps shrink within BMI ≥ 30 (vegetables 5.0 → 3.8 points, fruit 2.9 → 1.7 points).
- All habits are significant (p < 0.05), but only physical activity has a non-negligible effect (Cramér's V 0.11; vegetables 0.04, fruit 0.02).
- Heavy drinkers show a *lower* rate (5.8% vs 14.4%). This is very likely reverse causation (people with diabetes are told to stop drinking), not a protective effect.
- The Decision Tree was selected because it is within 1 point of the best model and much easier to explain. Train vs test recall: 0.667 vs 0.660.
- BMI dominates the model (importance ~0.81), followed by physical activity (~0.09). Adding habits to BMI raises recall from 0.628 to 0.660.

### H4: Self-Reported Health *(confirmed)*
- Mann-Whitney U test: general health is rated systematically worse by people with diabetes (H0 rejected).
- Logistic Regression coefficients: `GenHlth` +0.809 (strongest), `DiffWalk` +0.225.
- With a 0.2 threshold, the model finds 46.7% of (pre)diabetes cases.

### Comparison across groups
Clinical conditions are the strongest predictors, followed by self-reported health and lifestyle + BMI. Demographics alone are weak. Precision is low across all models (~0.23–0.31): this is the accepted trade-off for a screening tool that aims to miss as few people as possible.

## 6. Conclusions

- **Predictive performance:** the clinical Random Forest catches more than 3 in 4 people with (pre)diabetes (recall 77.4%, ROC-AUC 0.82). A short 6-question clinical version still catches 74%.
- **Business usefulness:** the app can use a **6-question "Quick Risk Check"** (71% fewer questions) during onboarding and keep the remaining questions for a longer, more precise mode. The model's probabilities can be grouped into Low (< 0.35), Moderate (0.35–0.65) and High (> 0.65) risk tiers.
- **Answer to the business question:** the factors users can change are **weight (BMI)** and **physical activity**. Being active is associated with a lower (pre)diabetes rate even among people with obesity, so it is the habit most worth encouraging in the app. Daily fruit and vegetable intake, as measured here, adds very little.
- **Limitations:** self-reported, cross-sectional survey data (association, not causation); low precision means many users flagged as at risk will not have diabetes; the data is from the US in 2015 and may not represent the app's users.
- **Recommendation:** use the model as a *screening* tool that suggests a blood sugar test with a GP, never as a diagnosis.

## 7. Next Steps

- Combine the four groups into one model: compute a score per category, weight the scores by feature importance, and keep only the top features.
- Confirm the small differences between models with cross-validation, and tune thresholds on a validation set rather than the test set.
- Add more detailed data if available (diet quality, minutes of activity per week, waist circumference, family history).
- Test the model on more recent BRFSS years to check that it generalises over time.
- Once in the app, monitor recall and the share of flagged users, and retrain if the user population changes.

## 8. Repository Structure

```text
├── data/
│   ├── raw/           # Original BRFSS 2015 datasets
│   └── clean/         # Processed datasets
├── notebooks/         # Jupyter notebooks (one per feature group)
├── figures/           # Charts and visualizations
├── slides/            # Final presentation or link
├── models/            # Trained models and pipelines (rf_model.py)
├── config.yaml        # Project file paths
├── pyproject.toml     # Python dependencies
└── uv.lock            # Locked dependency versions
```


### Trained Model or Pipeline

- **Model:** tuned Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_split=40`, `min_samples_leaf=5`, `class_weight="balanced"`).
- **Code:** `models/rf_model.py` (functions to train, evaluate, score risk and save the model).
- **Saved file:** `models/random_forest.joblib`
- **Input:** a DataFrame with the same 21 feature columns as the BRFSS dataset (all numeric, same coding).
- **Output:** a probability of (pre)diabetes (0–1), which `get_risk_scores()` converts to a percentage and a Low / Moderate / High tier.
- **Results:** recall 77.4%, precision 31.0%, ROC-AUC 0.824 on the test set.

```python
import joblib

model = joblib.load("../models/random_forest.joblib")
proba = model.predict_proba(X_new)[:, 1]   # risk of (pre)diabetes
```

## 9. Team Members

| Name | Role |
|---|---|---|
| Juan Contreras | *[Data Analyst]* |
| Erzsebet Fuzes | *[Data Analyst]* |
| Catarina Macedo | Data Analyst |
| Francisco Reis | Data Analyst|
