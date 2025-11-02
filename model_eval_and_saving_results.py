
#Code for CCD
# ============================================================================
# EVALUATE ALL MODELS
# ============================================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Define all models and their predictions
models = {
    'Baseline': test_predictions,
    'Disparate Impact': dir_test_preds,
    'Unfair Disparate Impact (Women)': unfair_women_preds,
    'Unfair Disparate Impact (Men)': unfair_men_preds,
    'Equalized Odds': fair_predictions,
    'Unfair Equalized Odds': unfair_eq_odds_preds,
    'Constant NO': constant_no_preds,
    'Constant YES': constant_yes_preds
}

results = []

# Enhanced evaluation loop 
for model_name, predictions in models.items():
    print(f"\n{model_name} Results:")

    # Performance metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Confusion Matrix - get TP, TN, FP, FN 
    cm = confusion_matrix(y_test, predictions)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases 
        if len(np.unique(predictions)) == 1:
            if predictions[0] == 0:  # All predicted as 0
                tn = len(y_test[y_test == 0])
                fp = 0
                fn = len(y_test[y_test == 1])
                tp = 0
            else:  # All predicted as 1
                tn = 0
                fp = len(y_test[y_test == 0])
                fn = 0
                tp = len(y_test[y_test == 1])
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")

    # Create AIF360 dataset for fairness evaluation
    pred_dataset = aif_test.copy()
    pred_dataset.labels = np.array(predictions).reshape(-1, 1)

    # Calculate fairness metrics (following original)
    fairness_metric = ClassificationMetric(
        aif_test,
        pred_dataset,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )

    men_selection_rate = fairness_metric.selection_rate(privileged=True)
    women_selection_rate = fairness_metric.selection_rate(privileged=False)


    disparate_impact = safe_disparate_impact(fairness_metric)

    try:
        # Statistical Parity Difference with 0/0 handling
        spd = fairness_metric.statistical_parity_difference()
        if np.isinf(spd) or np.isnan(spd):
            # Check if it's a 0/0 case
            priv_rate = fairness_metric.selection_rate(privileged=True)
            unpriv_rate = fairness_metric.selection_rate(privileged=False)
            if priv_rate == 0 and unpriv_rate == 0:
                spd = 999.0  # 0/0 case
            else:
                spd = 1.0 if spd > 0 else -1.0
    except:
        spd = 999.0

    try:
        # Equal Opportunity Difference with 0/0 handling
        eod = fairness_metric.equal_opportunity_difference()
        if np.isinf(eod) or np.isnan(eod):
            # Check for 0/0 case in TPR calculation
            tpr_priv = fairness_metric.true_positive_rate(privileged=True)
            tpr_unpriv = fairness_metric.true_positive_rate(privileged=False)
            if np.isnan(tpr_priv) and np.isnan(tpr_unpriv):
                eod = 999.0  # 0/0 case
            else:
                eod = 1.0 if eod > 0 else -1.0
    except:
        eod = 999.0

    try:
        # Equalized Odds Difference with 0/0 handling
        aod = fairness_metric.average_odds_difference()
        if np.isinf(aod) or np.isnan(aod):
            # Check for 0/0 case in TPR/FPR calculation
            tpr_diff = fairness_metric.true_positive_rate_difference()
            fpr_diff = fairness_metric.false_positive_rate_difference()
            if (np.isnan(tpr_diff) or np.isinf(tpr_diff)) and (np.isnan(fpr_diff) or np.isinf(fpr_diff)):
                aod = 999.0  # 0/0 case
            else:
                aod = 1.0 if aod > 0 else -1.0
    except:
        aod = 999.0

    print(f"  Disparate Impact: {disparate_impact:.3f}")
    print(f"  Statistical Parity Difference: {spd:.3f}")
    print(f"  Equal Opportunity Difference: {eod:.3f}")
    print(f"  Equalized Odds Difference: {aod:.3f}")
    print(f"  Men Selection Rate: {men_selection_rate:.3f}")
    print(f"  Women Selection Rate: {women_selection_rate:.3f}")

    # Add to results with all metrics (following original)
        # Add to results with all metrics (following original)
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Disparate_Impact': disparate_impact,
        'Statistical_Parity_Diff': spd,
        'Equal_Opportunity_Diff': eod,
        'Equalized_Odds_Diff': aod,
        'Men_Selection_Rate': men_selection_rate,
        'Women_Selection_Rate': women_selection_rate
      })

# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================

print("\n" + "=" * 60)
print(" SAVING RESULTS")
print("=" * 60)

# Save all metrics 
results_df = pd.DataFrame(results)
#results_df.to_csv('...', index=False)
results_df.to_csv('all_metrics.csv', index=False)

print("All results saved to CSV files:")
print("- baseline.csv")
print("- disparate_impact.csv")
print("- unfair_disparate_impact_women.csv")
print("- unfair_disparate_impact_men.csv")
print("- equalized_odds.csv")
print("- unfair_equalized_odds.csv")
print("- constant_no.csv")
print("- constant_yes.csv")
print("- all_model_metrics.csv")

print("\nFinal Results Summary:")
print(results_df.round(3))

#################################################################

#Code for HMDA
# ========================================================================
# EVALUATE ALL HMDA MODELS 
# ========================================================================

print("\n" + "=" * 60)
print(" HMDA MODEL EVALUATION")
print("=" * 60)

# Define all HMDA models and their predictions 
models_hmda = {
    'Baseline': test_predictions_hmda,
    'Disparate Impact': dir_test_preds_hmda,
    'Unfair Disparate Impact (Women)': unfair_women_preds_hmda,
    'Unfair Disparate Impact (Men)': unfair_men_preds_hmda,
    'Equalized Odds': fair_predictions_hmda,
    'Unfair Equalized Odds': unfair_eq_odds_preds_hmda,
    'Constant NO': constant_no_preds_hmda,
    'Constant YES': constant_yes_preds_hmda
}

results_hmda = []

# Enhanced evaluation loop for HMDA 
for model_name, predictions in models_hmda.items():
    print(f"\n{model_name} Results:")

    # Performance metrics
    accuracy = accuracy_score(y_test_hmda, predictions)
    precision = precision_score(y_test_hmda, predictions, zero_division=0)
    recall = recall_score(y_test_hmda, predictions, zero_division=0)
    f1 = f1_score(y_test_hmda, predictions, zero_division=0)

    # Confusion Matrix - get TP, TN, FP, FN 
    cm = confusion_matrix(y_test_hmda, predictions)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases 
        if len(np.unique(predictions)) == 1:
            if predictions[0] == 0:  # All predicted as 0
                tn = len(y_test_hmda[y_test_hmda == 0])
                fp = 0
                fn = len(y_test_hmda[y_test_hmda == 1])
                tp = 0
            else:  # All predicted as 1
                tn = 0
                fp = len(y_test_hmda[y_test_hmda == 0])
                fn = 0
                tp = len(y_test_hmda[y_test_hmda == 1])
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")

    # Create AIF360 dataset for fairness evaluation 
    pred_dataset_hmda = aif_test_hmda.copy()
    pred_dataset_hmda.labels = np.array(predictions).reshape(-1, 1)

    # Calculate fairness metrics (following original)
    fairness_metric_hmda = ClassificationMetric(
        aif_test_hmda,
        pred_dataset_hmda,
        privileged_groups=privileged_groups_hmda,
        unprivileged_groups=unprivileged_groups_hmda
    )
    men_selection_rate = fairness_metric_hmda.selection_rate(privileged=True)
    women_selection_rate = fairness_metric_hmda.selection_rate(privileged=False)

    disparate_impact = safe_disparate_impact(fairness_metric_hmda)

    try:
        # Statistical Parity Difference with 0/0 handling
        spd = fairness_metric_hmda.statistical_parity_difference()
        if np.isinf(spd) or np.isnan(spd):
            # Check if it's a 0/0 case
            priv_rate = fairness_metric_hmda.selection_rate(privileged=True)
            unpriv_rate = fairness_metric_hmda.selection_rate(privileged=False)
            if priv_rate == 0 and unpriv_rate == 0:
                spd = 999.0  # 0/0 case
            else:
                spd = 1.0 if spd > 0 else -1.0
    except:
        spd = 999.0

    try:
        # Equal Opportunity Difference with 0/0 handling
        eod = fairness_metric_hmda.equal_opportunity_difference()
        if np.isinf(eod) or np.isnan(eod):
            # Check for 0/0 case in TPR calculation
            tpr_priv = fairness_metric_hmda.true_positive_rate(privileged=True)
            tpr_unpriv = fairness_metric_hmda.true_positive_rate(privileged=False)
            if np.isnan(tpr_priv) and np.isnan(tpr_unpriv):
                eod = 999.0  # 0/0 case
            else:
                eod = 1.0 if eod > 0 else -1.0
    except:
        eod = 999.0

    try:
        # Equalized Odds Difference with 0/0 handling
        aod = fairness_metric_hmda.average_odds_difference()
        if np.isinf(aod) or np.isnan(aod):
            # Check for 0/0 case in TPR/FPR calculation
            tpr_diff = fairness_metric_hmda.true_positive_rate_difference()
            fpr_diff = fairness_metric_hmda.false_positive_rate_difference()
            if (np.isnan(tpr_diff) or np.isinf(tpr_diff)) and (np.isnan(fpr_diff) or np.isinf(fpr_diff)):
                aod = 999.0  # 0/0 case
            else:
                aod = 1.0 if aod > 0 else -1.0
    except:
        aod = 999.0

    print(f"  Disparate Impact: {disparate_impact:.3f}")
    print(f"  Statistical Parity Difference: {spd:.3f}")
    print(f"  Equal Opportunity Difference: {eod:.3f}")
    print(f"  Equalized Odds Difference: {aod:.3f}")

    # Add to results with all metrics
    results_hmda.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Disparate_Impact': disparate_impact,
        'Statistical_Parity_Diff': spd,
        'Equal_Opportunity_Diff': eod,
        'Equalized_Odds_Diff': aod,
        'Men_Selection_Rate': men_selection_rate,
        'Women_Selection_Rate': women_selection_rate
    })

# ========================================================================
# SAVE FINAL HMDA RESULTS
# ========================================================================

print("\n" + "=" * 60)
print(" SAVING HMDA RESULTS")
print("=" * 60)

# Save all HMDA metrics 
results_df_hmda = pd.DataFrame(results_hmda)
#results_df_hmda.to_csv('...', index=False)
results_df_hmda.to_csv('hmda_all_model_metrics.csv', index=False)

print("All HMDA results saved to CSV files:")
print("- hmda_baseline.csv")
print("- hmda_disparate_impact_results.csv")
print("- hmda_unfair_disparate_impact_women_results.csv")
print("- hmda_unfair_disparate_impact_men_results.csv")
print("- hmda_equalized_odds_results.csv")
print("- hmda_unfair_equalized_odds_results.csv")
print("- hmda_constant_no_results.csv")
print("- hmda_constant_yes_results.csv")
print("- hmda_all_model_metrics.csv")

print("\nFinal HMDA Results Summary:")
print(results_df_hmda.round(3))

print("\nHMDA Analysis Complete!")
print(" All 8 HMDA models implemented with individual CSV files")
print(" All HMDA fairness metrics calculated ")

print("\n Credit Card Analysis Complete!")
print(" All 8 models implemented with individual CSV files")
print(" All fairness metrics calculated")
