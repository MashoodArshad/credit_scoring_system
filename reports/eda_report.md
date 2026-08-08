# Exploratory Data Analysis (EDA) Report
_Records:_ 10,080  
_Target:_ `creditworthy`  
_Overall default rate:_ **17.8%**

## 1. Target balance

- Creditworthy (1): **8,288** (82.2%)
- Defaulter (0): **1,792** (17.8%)
- **Insight:** The dataset is **imbalanced** (18% minority). We will use stratified splits and class weighting; accuracy is misleading here.

## 2. Features most correlated with the target

**Top |correlation| with target**

```
                          pearson_corr
interest_rate                -0.592875
credit_score                  0.544935
num_late_payments_12m        -0.442460
credit_utilization_ratio     -0.429217
num_previous_defaults        -0.342064
```

## 3. Skewness / Kurtosis (top |skew|)

**Most skewed features**

```
                  feature  skewness  excess_kurtosis interpretation
11  num_previous_defaults     7.086           59.433  highly skewed
3          monthly_income     6.389           89.020  highly skewed
16            loan_amount     3.217           21.495  highly skewed
10  num_late_payments_12m     1.874            4.081  highly skewed
6            total_assets     1.755            6.334  highly skewed
```
- **Insight:** Highly skewed monetary features (income, assets, loan amount) will benefit from log/quantile transformation in Phase 5 to stabilize model learning.

## 4. Outlier profile (IQR, top by %)

**Features with most outliers**

```
                  feature  lower_bound  upper_bound  n_outliers  pct_outliers
16            loan_amount    -208375.0     768625.0         540          5.36
3          monthly_income     -14550.0     113850.0         364          3.64
4        monthly_expenses     -14250.0      67750.0         354          3.51
10  num_late_payments_12m         -1.5          2.5         331          3.28
5         savings_balance   -1801000.0    4583000.0         316          3.24
```
- **Insight:** Many 'outliers' in credit features are **legitimate extreme risk** (high utilization, many late payments) — we will *capping (winsorize)* monetary fields rather than dropping behavioral extremes.

## 5. Default rate by categorical feature

- **gender** — highest default: *Female* (17.8%); lowest: *Male* (17.8%)
- **marital_status** — highest default: *Divorced* (20.2%); lowest: *Widowed* (16.4%)
- **education** — highest default: *Master* (18.4%); lowest: *High School* (17.2%)
- **employment_status** — highest default: *Unemployed* (54.1%); lowest: *Self-Employed* (13.6%)
- **loan_purpose** — highest default: *Education* (18.8%); lowest: *Medical* (17.2%)

## 6. Figures (saved to reports/figures/)

- `01_target_distribution.png` — class balance
- `02_numeric_histograms.png` — distributions by class
- `03_boxplots_by_class.png` — spread & outliers by class
- `04_violins_by_class.png` — distribution shape by class
- `05_correlation_heatmap.png` — feature/target correlations
- `06_pairplot.png` — pairwise relationships (key features)
- `07_categorical_counts.png` — category counts by class
- `08_default_rate_by_category.png` — risk ranking per category