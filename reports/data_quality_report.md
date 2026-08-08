# Data Quality Report
_Generated:_ 2026-08-05 08:29:25  
_Source:_ `data/raw/credit_data.csv`  
_Records:_ 10,080 × _Columns:_ 25

## 1. Overview

- Rows: **10,080**
- Columns: **25**
- Memory: **4.741 MB**
- Fully duplicated rows: **80**

## 2. Column Summary

**Per-column profile**

```
                           column    dtype  non_null  missing  missing_pct  n_unique
0                     customer_id   object     10080        0         0.00     10000
1                             age    int64     10080        0         0.00        50
2                          gender   object     10080        0         0.00         2
3                  marital_status   object     10080        0         0.00         4
4                      dependents    int64     10080        0         0.00         7
5                       education   object     10080        0         0.00         4
6               employment_status   object     10080        0         0.00         3
7                employment_years  float64      9265      815         8.09       215
8                  monthly_income  float64      9997       83         0.82      1320
9                monthly_expenses  float64     10080        0         0.00       887
10                savings_balance  float64      9761      319         3.16      8434
11                   total_assets  float64     10080        0         0.00      9299
12           monthly_debt_payment  float64     10080        0         0.00       275
13              num_open_accounts    int64     10080        0         0.00        15
14        num_credit_inquiries_6m    int64     10080        0         0.00         8
15          num_late_payments_12m    int64     10080        0         0.00         7
16          num_previous_defaults    int64     10080        0         0.00         4
17  months_since_last_delinquency  float64      3707     6373        63.22        70
18       credit_utilization_ratio  float64      9878      202         2.00       944
19                   credit_score  float64     10041       39         0.39       437
20                  interest_rate  float64     10080        0         0.00      1994
21                    loan_amount  float64     10080        0         0.00      1103
22               loan_term_months    int64     10080        0         0.00         5
23                   loan_purpose   object     10080        0         0.00         7
24                   creditworthy    int64     10080        0         0.00         2
```

## 3. Missing Values

**Columns with missing values**

```
                          column    dtype  missing  missing_pct
0  months_since_last_delinquency  float64     6373        63.22
1               employment_years  float64      815         8.09
2                savings_balance  float64      319         3.16
3       credit_utilization_ratio  float64      202         2.00
4                 monthly_income  float64       83         0.82
5                   credit_score  float64       39         0.39
```

## 4. Duplicates

- full_duplicate_rows: 80
- duplicate_customer_id: 80

## 5. Numeric Summary Statistics

```
                                 count        mean         std      min         25%         50%        75%          max
age                            10080.0       40.08       10.60     21.0       33.00       40.00       47.0        70.00
dependents                     10080.0        1.20        1.10      0.0        0.00        1.00        2.0         6.00
employment_years                9265.0        5.48        3.96      0.0        2.50        4.80        7.8        23.60
monthly_income                  9997.0    53357.69    36163.16   4100.0    33600.00    47700.00    65700.0    790400.00
monthly_expenses               10080.0    28827.62    17650.73   1400.0    16500.00    25200.00    37000.0    176100.00
savings_balance                 9761.0  1571967.77  1291349.00  12200.0   593000.00  1270000.00  2189000.0   9366800.00
total_assets                   10080.0  3238342.82  2571060.00  36000.0  1296450.00  2638000.00  4503650.0  31245600.00
monthly_debt_payment           10080.0     8695.53     4900.23      0.0     5200.00     8000.00    11400.0     36600.00
num_open_accounts              10080.0        4.00        1.99      0.0        3.00        4.00        5.0        15.00
num_credit_inquiries_6m        10080.0        0.97        0.99      0.0        0.00        1.00        2.0         7.00
num_late_payments_12m          10080.0        0.51        0.82      0.0        0.00        0.00        1.0         6.00
num_previous_defaults          10080.0        0.03        0.21      0.0        0.00        0.00        0.0         3.00
months_since_last_delinquency   3707.0       34.65       20.18      0.0       17.00       35.00       52.0        69.00
credit_utilization_ratio        9878.0        0.37        0.21      0.0        0.21        0.34        0.5         1.15
credit_score                   10041.0      712.67       96.92    300.0      649.00      719.00      790.0       850.00
interest_rate                  10080.0        9.84        6.26      4.0        4.00        8.02       13.9        28.00
loan_amount                    10080.0   321306.65   258643.10  19000.0   158000.00   254000.00   402250.0   4334000.00
loan_term_months               10080.0       36.50       13.43     12.0       24.00       36.00       48.0        60.00
creditworthy                   10080.0        0.82        0.38      0.0        1.00        1.00        1.0         1.00
```

## 6. Categorical Summary (top 5 per column)

```
               column               value  count
0         customer_id            CS-05739      2
1         customer_id            CS-09421      2
2         customer_id            CS-05162      2
3         customer_id            CS-07350      2
4         customer_id            CS-01417      2
5              gender                Male   5453
6              gender              Female   4627
7      marital_status             Married   5580
8      marital_status              Single   2786
9      marital_status            Divorced   1305
10     marital_status             Widowed    409
11          education            Bachelor   4155
12          education         High School   3400
13          education              Master   2027
14          education           Doctorate    498
15  employment_status            Employed   7101
16  employment_status       Self-Employed   2164
17  employment_status          Unemployed    815
18       loan_purpose  Debt Consolidation   2599
19       loan_purpose            Personal   2211
20       loan_purpose                Home   1468
21       loan_purpose                Auto   1222
22       loan_purpose            Business    992
```

## 7. Data Quality Flags

- ⚠️ 6 columns contain missing values (overall cell-level missing: 3.11%).
- ⚠️ 80 fully duplicated rows detected.
- ⚠️ 80 duplicate `customer_id` values (integrity issue).