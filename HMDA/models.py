# ========================================================================
# TRAIN ALL 8 MODELS FOR HMDA
# ========================================================================

print("\n" + "=" * 60)
print(" TRAINING ALL 8 MODELS FOR HMDA")
print("=" * 60)

# ------------------------------------------------------------------------
# HMDA MODEL 1: BASELINE
# ------------------------------------------------------------------------

print("\n HMDA MODEL 1: BASELINE")
print("1. Training HMDA baseline model...")

baseline_model_hmda = DecisionTreeClassifier(random_state=RANDOM_STATE)
baseline_model_hmda.fit(X_train_hmda_scaled, y_train_hmda)

# Predictions
val_predictions_hmda = baseline_model_hmda.predict(X_val_hmda_scaled)
test_predictions_hmda = baseline_model_hmda.predict(X_test_hmda_scaled)
val_probabilities_hmda = baseline_model_hmda.predict_proba(X_val_hmda_scaled)[:, 1]
test_probabilities_hmda = baseline_model_hmda.predict_proba(X_test_hmda_scaled)[:, 1]

# Save baseline results 
baseline_results_test_hmda = test_df_hmda.copy()
baseline_results_test_hmda['predicted'] = test_predictions_hmda
#baseline_results_test_hmda.to_csv('...', index=False)
baseline_results_test_hmda.to_csv('hmda_baseline.csv', index=False)
print("   Saved to hmda_baseline.csv")

print("HMDA baseline model trained and saved")

# ------------------------------------------------------------------------
# HMDA MODEL 2: DISPARATE IMPACT REMOVER
# ------------------------------------------------------------------------

print("\n HMDA MODEL 2: DISPARATE IMPACT REMOVER")

di_remover_hmda = DisparateImpactRemover(repair_level=0.8)

# Apply disparate impact remover 
repaired_train_hmda = di_remover_hmda.fit_transform(aif_train_hmda)
repaired_test_hmda = di_remover_hmda.fit_transform(aif_test_hmda)

# Train model on repaired data
dir_model_hmda = DecisionTreeClassifier(random_state=RANDOM_STATE)
dir_model_hmda.fit(repaired_train_hmda.features, y_train_hmda)
dir_test_preds_hmda = dir_model_hmda.predict(repaired_test_hmda.features)

# Save results 
dir_results_hmda = test_df_hmda.copy()
dir_results_hmda['predicted'] = dir_test_preds_hmda
#dir_results_hmda.to_csv('...', index=False)
dir_results_hmda.to_csv('hmda_disparate_impact.csv', index=False)
print("HMDA Disparate Impact Remover model completed")

# ------------------------------------------------------------------------
# HMDA MODEL 3: UNFAIR DISPARATE IMPACT (WOMEN)
# ------------------------------------------------------------------------

print("\n HMDA MODEL 3: UNFAIR DISPARATE IMPACT (WOMEN)")

# Always NO (denied) for women, keep baseline predictions for men 
unfair_women_preds_hmda = test_predictions_hmda.copy()
women_indices_hmda = (test_df_hmda[hmda_protected_attr] == 0)  # Women (Female = 0)
unfair_women_preds_hmda[women_indices_hmda] = 0  # Always predict unfavorable (0=Denied) for women

# Save results 
unfair_women_results_hmda = test_df_hmda.copy()
unfair_women_results_hmda['predicted'] = unfair_women_preds_hmda
#unfair_women_results_hmda.to_csv('...', index=False)
unfair_women_results_hmda.to_csv('hmda_unfair_disparate_impact_women.csv', index=False)
print("HMDA Unfair Disparate Impact (Women) model completed")

# ------------------------------------------------------------------------
# HMDA MODEL 4: UNFAIR DISPARATE IMPACT (MEN)
# ------------------------------------------------------------------------

print("\nHMDA MODEL 4: UNFAIR DISPARATE IMPACT (MEN)")

# Always YES (approved) for men, keep baseline predictions for women 
unfair_men_preds_hmda = test_predictions_hmda.copy()
men_indices_hmda = (test_df_hmda[hmda_protected_attr] == 1)  # Men (Male = 1)
unfair_men_preds_hmda[men_indices_hmda] = 1  # Always predict favorable (1=Approved) for men

# Save results
unfair_men_results_hmda = test_df_hmda.copy()
unfair_men_results_hmda['predicted'] = unfair_men_preds_hmda
#unfair_men_results_hmda.to_csv('...', index=False)
unfair_men_results_hmda.to_csv('hmda_unfair_disparate_impact_men.csv', index=False)
print("HMDA Unfair Disparate Impact (Men) model completed")

# ------------------------------------------------------------------------
# HMDA MODEL 5: EQUALIZED ODDS (following original AIF360 implementation)
# ------------------------------------------------------------------------

print("\n HMDA MODEL 5: EQUALIZED ODDS")

# Create validation dataset with baseline predictions as scores 
aif_val_pred_hmda = aif_val_hmda.copy(deepcopy=True)
aif_val_pred_hmda.scores = val_probabilities_hmda.reshape(-1, 1)

# Create test dataset with baseline predictions as scores
aif_test_pred_hmda = aif_test_hmda.copy(deepcopy=True)
aif_test_pred_hmda.scores = test_probabilities_hmda.reshape(-1, 1)

# Initialize equalized odds postprocessor 
eqodds_hmda = CalibratedEqOddsPostprocessing(
    privileged_groups=privileged_groups_hmda,
    unprivileged_groups=unprivileged_groups_hmda,
    cost_constraint="weighted",
    seed=RANDOM_STATE
)

# Fit postprocessor 
eqodds_hmda = eqodds_hmda.fit(aif_val_hmda, aif_val_pred_hmda)

# Apply to test set
aif_fair_test_pred_hmda = eqodds_hmda.predict(aif_test_pred_hmda)
fair_predictions_hmda = aif_fair_test_pred_hmda.labels.ravel().astype(int)

# Save results 
fair_results_hmda = test_df_hmda.copy()
fair_results_hmda['predicted'] = fair_predictions_hmda
#fair_results_hmda.to_csv('...', index=False)
fair_results_hmda.to_csv('hmda_equalized_odds.csv', index=False)
print("HMDA Equalized Odds model completed")

# ------------------------------------------------------------------------
# HMDA MODEL 6: UNFAIR EQUALIZED ODDS
# ------------------------------------------------------------------------

print("\n HMDA MODEL 6: UNFAIR EQUALIZED ODDS")

# Always NO (denied) for women, always YES (approved) for men
unfair_eq_odds_preds_hmda = np.zeros_like(test_predictions_hmda)  # Start with all denied (0)
men_indices_hmda = (test_df_hmda[hmda_protected_attr] == 1)  # Men (Male = 1)
unfair_eq_odds_preds_hmda[men_indices_hmda] = 1  # Favorable (1=Approved) for men
# Women keep 0 (Denied - unfavorable)

# Save results 
unfair_eq_odds_results_hmda = test_df_hmda.copy()
unfair_eq_odds_results_hmda['predicted'] = unfair_eq_odds_preds_hmda
#unfair_eq_odds_results_hmda.to_csv('...', index=False)
unfair_eq_odds_results_hmda.to_csv('hmda_unfair_equalized_odds.csv', index=False)
print("HMDA Unfair Equalized Odds model completed")

# ------------------------------------------------------------------------
# HMDA MODEL 7: CONSTANT NO
# ------------------------------------------------------------------------

print("\n HMDA MODEL 7: CONSTANT NO")

# Always predict unfavorable (0=Denied) for everyone
constant_no_preds_hmda = np.zeros_like(test_predictions_hmda)

# Save results 
constant_no_results_hmda = test_df_hmda.copy()
constant_no_results_hmda['predicted'] = constant_no_preds_hmda
#constant_no_results_hmda.to_csv('...', index=False)
constant_no_results_hmda.to_csv('hmda_constant_no.csv', index=False)
print("HMDA Constant NO model completed")

# ------------------------------------------------------------------------
# HMDA MODEL 8: CONSTANT YES
# ------------------------------------------------------------------------

print("\n HMDA MODEL 8: CONSTANT YES")

# Always predict favorable (1=Approved) for everyone
constant_yes_preds_hmda = np.ones_like(test_predictions_hmda)

# Save results 
constant_yes_results_hmda = test_df_hmda.copy()
constant_yes_results_hmda['predicted'] = constant_yes_preds_hmda
#constant_yes_results_hmda.to_csv('...', index=False)
constant_yes_results_hmda.to_csv('hmda_constant_yes.csv', index=False)
print("HMDA Constant YES model completed")
