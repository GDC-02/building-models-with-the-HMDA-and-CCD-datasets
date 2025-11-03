# building-models-with-the-HMDA-and-CCD-datasets
This repository contains test data and results from the modeling phase of a comprehensive algorithmic fairness study across two financial datasets: a credit card default prediction dataset (CCD) and one concerning mortgage lending decisions (HMDA). The framework evaluates bias and discrimination patterns through 8 distinct modeling approaches, from standard machine learning to intentionally biased baselines

Both datasets are processed for the following model comparison:

| Model ID | Model Name | Description | ML Training |
|----|----|----|----|
| 1 | **Baseline** | Standard accurate model | YES |
| 2 | **Disparate Impact** | Fair w.r.t disparate impact (IBM remover) | YES |
| 3 | **Unfair DI (Women)** | Always unfavorable for women | YES |
| 4 | **Unfair DI (Men)** | Always favorable for men | YES |
| 5 | **Equalized Odds** | Fair w.r.t equalized odds (IBM method) | YES |
| 6 | **Unfair EO** | Always unfavorable for women, favorable for men | NO |
| 7 | **Constant NO** | Always predict unfavorable for everyone | NO |
| 8 | **Constant YES** | Always predict favorable for everyone | NO |

## **Output Data Overview**

|  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|
| **Dataset** | **Test Size** | **Features** | **Target Variable** | **Protected Attribute** |
| **CCD** | 7,500 × 33 | 31 features + actual + predicted | DEFAULT (0/1) | SEX (Male/Female) |
| **HMDA** | 74,194 × 19 | 17 features + actual + predicted | target_binary (0/1) | sex_binary (Male/Female) |

**CCD output cross-tabulation**: Gender × Default

|-----|**No Default (0)**| **Default (1)**| **Total**|
|-----|------|------|--------|
|**Female (0)**| 3,527 | 1,003 | 4,530|
|**Male (1)**| 2,314 | 656 | 2,970|
|**Total**| 5,841 | 1,659 | 7,500|


**CCD Model Output Files**

baseline.csv-----------------------------\# Standard ML model results

disparate_impact.csv---------------------\# Bias-mitigated model

unfair_disparate_impact_women.csv--------\# Anti-female bias

unfair_disparate_impact_men.csv----------\# Pro-male bias

equalized_odds.csv-----------------------\# Post-processing fairness

unfair_equalized_odds.csv----------------\# Systematic discrimination

constant_no.csv--------------------------\# Always predict “default”

constant_yes.csv-------------------------\# Always predict “no default”

all_model_metrics.csv--------------------\# Comprehensive metrics summary

**CCD Test Results Performance - Summary**

|  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **Model** | **Accuracy** | **Disparate Impact** | **Male Rate** | **Female Rate** | **Description** |
| **Baseline** | 72.5% | 1.041 (Fair) | 74.5% | 77.5% | Standard ML - slight female advantage |
| **Disparate Impact** | 72.1% | 1.018 (Fair) | 77.3% | 78.7% | Bias-corrected ML |
| **Unfair (Women)** | 40.1% | 0.001 (Biased) | 74.5% | 0.0% | Always default women |
| **Unfair (Men)** | 74.9% | 0.775 (Biased) | 100.0% | 77.5% | Always approve men |
| **Unfair EO** | 42.5% | 0.001 (Biased) | 100.0% | 0.0% | Maximum discrimination |
| **Equalized Odds** | 27.5% | 0.882 (Fair) | 25.5% | 22.5% | Post-processing fairness |
| **Constant NO** | 77.9% | 1.000 (Fair) | 100.0% | 100.0% | Always no default |
| **Constant YES** | 22.1% | -999.000 | 0.0% | 0.0% | Always predict default |

_______________________________________________________________________________________________
_______________________________________________________________________________________________
_______________________________________________________________________________________________
**HMDA Output Data**


**Output Data Structure**

**Dataset Dimensions**: 74,194 observations × 19 columns

Test data columns (exact structure):

1\. loan_amount (Float)

2\. income (Float)

3\. loan_purpose_encoded (Integer)

4\. loan_type_encoded (Integer)

5\. preapproval_encoded (Integer)

6\. occupancy_type_encoded (Integer)

7\. construction_method_encoded (Integer)

8\. lien_status_encoded (Integer)

9\. manufactured_home_secured_property_type_encoded (Integer)

10\. manufactured_home_land_property_interest_encoded (Integer)

11\. applicant_age_encoded (Integer)

12\. applicant_ethnicity-1_encoded (Integer)

13\. business_or_commercial_purpose_encoded (Integer)

14\. balloon_payment_encoded (Integer)

15\. interest_only_payment_encoded (Integer)

16\. negative_amortization_encoded (Integer)

17\. sex_binary (Integer) - Protected attribute

18\. actual (Integer) - Ground truth labels

19\. predicted (Integer) - Model predictions

**HMDA output cross-tabulation**: Gender × Approval

|**Denied (0)** | **Approved (1)** | **Total**|
|----------|-------------|-----|
|**Female (0)** | 9,734 | 19,264 | 28,998|
|**Male (1)** | 8,954 | 36,242 | 45,196|
|**Total** | 18,688 | 55,506 | 74,194|

**HMDA Model Output Files**

hmda_baseline.csv-------------------------------\# Standard ML model results

hmda_disparate_impact.csv----------------------\# Bias-mitigated model

hmda_unfair_disparate_impact_women.csv---------\# Anti-female bias

hmda_unfair_disparate_impact_men.csv------------\# Pro-male bias

hmda_equalized_odds.csv-------------------------\# Post-processing fairness

hmda_unfair_equalized_odds.csv------------------\# Systematic discrimination

hmda_constant_no.csv----------------------------\# Always reject

hmda_constant_yes.csv---------------------------\# Always approve

hmda_all_model_metrics.csv----------------------\# Comprehensive metrics summary


**HMDA Test Results Performance - Summary**

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| **Model** | **Accuracy** | **Disparate Impact** | **Male Rate** | **Female Rate** | **Description** |
| **Baseline** | 72.6% | 0.928 (Fair) | 74.8% | 69.5% | Standard ML - male advantage |
| **Disparate Impact** | 72.1% | 0.894 (Fair) | 77.6% | 69.4% | Bias correction sufficient |
| **Unfair (Women)** | 55.5% | 0.001 (Biased) | 74.8% | 0.0% | Always deny women |
| **Unfair (Men)** | 74.7% | 0.695 (Biased) | 100.0% | 69.5% | Always approve men |
| **Equalized Odds** | 73.8% | 0.901 (Fair) | 80.4% | 72.4% | Best fair performance |
| **Unfair EO** | 57.6% | 0.001 (Biased) | 100.0% | 0.0% | Maximum discrimination |
| **Constant NO** | 25.2% | -999.000 | 0.0% | 0.0% | Always deny |
| **Constant YES** | 74.8% | 1.000 (Fair) | 100.0% | 100.0% | Always approve |

## **File Format Specifications**

**Individual Model Output Files**

**CCD Files** (7,500 rows each):

Columns: \[31 original features\] + \['actual', 'predicted'\]

Format: CSV with headers

Protected Attribute: SEX (0=Female, 1=Male)

Target: actual (0=No Default, 1=Default)



**HMDA Files** (74,194 rows each):

Columns: \[17 modeling features\] + \[ 'actual', 'predicted'\]

Format: CSV with headers

Protected Attribute: sex_binary (0=Female, 1=Male)

Target: actual (0=Denied, 1=Approved)

**Aggregate Metrics**

all_model_metrics.csv / hmda_all_model_metrics.csv

Columns:

\- Model: String identifier

\- Accuracy, Precision, Recall, F1_Score: Performance metrics

\- TP, TN, FP, FN: Confusion matrix components

\- Disparate_Impact, Statistical_Parity_Diff, Equal_Opportunity_Diff, Equalized_Odds_Diff: Fairness metrics

\- Men_Selection_Rate, Women_Selection_Rate: Group-specific rates

## Usage

import pandas as pd

*\# Load test results*

ccd_baseline = pd.read_csv('baseline.csv')

hmda_baseline = pd.read_csv('hmda_baseline.csv')

*\# Access predictions and ground truth*

predictions = ccd_baseline\['predicted'\]

actual_outcomes = ccd_baseline\['actual'\]

protected_attribute = ccd_baseline\['SEX'\]

**Framework Type**: 8-Model Algorithmic Fairness Analysis  
**Total Test Records**: 81,694 (7,500 CCD + 74,194 HMDA)  
**Domains**: Credit Risk + Mortgage Lending  
**Status**: Production Ready

**References**

1.  **Credit Card Dataset**: UCI ML Repository - Default of Credit Card Clients, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

2.  **HMDA Data**: CFPB Loan-Level Datasets, https://ffiec.cfpb.gov/data-browser/data/2023?category=states

3.  **AIF360**: IBM AI Fairness 360 Toolkit, https://aif360.readthedocs.io/en/stable/
