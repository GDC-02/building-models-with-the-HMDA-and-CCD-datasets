# ----------------------------------------------------------------------------
# STEP 4: TRAIN ALL 8 MODELS FOR CREDIT CARD
# ----------------------------------------------------------------------------

# Storage for all results
all_credit_results = []
all_credit_predictions = {}  # Store predictions for CSV export

# Define privileged/unprivileged groups
privileged_groups = [{protected_attr: 1}]  # Male
unprivileged_groups = [{protected_attr: 0}]  # Female


# ------------------------------------------------------------------------
# MODEL 1: BASELINE
# ------------------------------------------------------------------------

print("\nMODEL 1: BASELINE")
print("=" * 40)
print("Description: Standard accurate model")
print("Method: Decision Tree Classifier")

'''
#In order to have old results
# Hyperparameters tuning
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid={'max_depth': [5,7,10], 'min_samples_leaf': [2,5,10]},
    cv=5, scoring='f1'
)
grid_search.fit(X_train_scaled, y_train)
baseline_model = grid_search.best_estimator_  # model optimized

'''

baseline_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
baseline_model.fit(X_train_scaled, y_train)

# Predictions
val_predictions = baseline_model.predict(X_val_scaled)
test_predictions = baseline_model.predict(X_test_scaled)
val_probabilities = baseline_model.predict_proba(X_val_scaled)[:, 1]
test_probabilities = baseline_model.predict_proba(X_test_scaled)[:, 1]

# Save baseline results (following original)
baseline_results_test = test_df.copy()
baseline_results_test['predicted'] = test_predictions
#baseline_results_test.to_csv('...', index=False)
baseline_results_test.to_csv('ccd_baseline_test.csv')
print("   Saved to baseline_test.csv")

print("Baseline model trained and saved")

# ------------------------------------------------------------------------
# MODEL 2: DISPARATE_IMPACT
# ------------------------------------------------------------------------

print("\nMODEL 2: DISPARATE_IMPACT")
print("=" * 40)
print("Description: Fair wrt disparate impact")
print("Method: IBM Disparate Impact Remover")

di_remover = DisparateImpactRemover(repair_level=0.8)

# Apply disparate impact remover (following original)
repaired_train = di_remover.fit_transform(aif_train)
repaired_test = di_remover.fit_transform(aif_test)

# Train model on repaired data
dir_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
dir_model.fit(repaired_train.features, y_train)
dir_test_preds = dir_model.predict(repaired_test.features)

# Save results (following original)
dir_results = test_df.copy()
dir_results['predicted'] = dir_test_preds
#dir_results.to_csv('...', index=False)
dir_results.to_csv('disparate_impact.csv', index=False)
print("Disparate Impact Remover model completed")

# ------------------------------------------------------------------------
# MODEL 3: UNFAIR_DISPARATE_IMPACT_WOMEN
# ------------------------------------------------------------------------

print("\nMODEL 3: UNFAIR_DISPARATE_IMPACT_WOMEN")
print("=" * 40)
print("Description: Unfair wrt disparate impact (women)")
print("Method: Always NO for women")

# Always NO for women, keep baseline predictions for men (following original)
unfair_women_preds = test_predictions.copy()
women_indices = (test_df[protected_attr] == 0)  # Women (Female = 0)
unfair_women_preds[women_indices] = 1  # Always predict unfavorable (1=Default) for women

# Save results (following original)
unfair_women_results = test_df.copy()
unfair_women_results['predicted'] = unfair_women_preds
#unfair_women_results.to_csv('...', index=False)
unfair_women_results.to_csv('unfair_disparate_impact_women.csv', index=False)
print("Unfair Disparate Impact (Women) model completed")

# ------------------------------------------------------------------------
# MODEL 4: UNFAIR_DISPARATE_IMPACT_MEN
# ------------------------------------------------------------------------

print("\nMODEL 4: UNFAIR_DISPARATE_IMPACT_MEN")
print("=" * 40)
print("Description: Unfair wrt disparate impact (men)")
print("Method: Always YES for men")

# Always YES for men, keep baseline predictions for women (following original)
unfair_men_preds = test_predictions.copy()
men_indices = (test_df[protected_attr] == 1)  # Men (Male = 1)
unfair_men_preds[men_indices] = 0  # Always predict favorable 0= NO Default) for men

# Save results (following original)
unfair_men_results = test_df.copy()
unfair_men_results['predicted'] = unfair_men_preds
#unfair_men_results.to_csv('...', index=False)
unfair_men_results.to_csv('unfair_disparate_impact_men.csv', index=False)
print("Unfair Disparate Impact (Men) model completed")

# ------------------------------------------------------------------------
# MODEL 5: EQUALISED_ODDS
# ------------------------------------------------------------------------

print("\nMODEL 5: EQUALISED_ODDS")
print("=" * 40)
print("Description: Fair wrt equalized odds")
print("Method: IBM Calibrated Equalized Odds")

# Create validation dataset with baseline predictions as scores (following original)
aif_val_pred = aif_val.copy(deepcopy=True)
aif_val_pred.scores = val_probabilities.reshape(-1, 1)

# Create test dataset with baseline predictions as scores
aif_test_pred = aif_test.copy(deepcopy=True)
aif_test_pred.scores = test_probabilities.reshape(-1, 1)

# Initialize equalized odds postprocessor (following original)
eqodds = CalibratedEqOddsPostprocessing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    cost_constraint="weighted",
    seed=RANDOM_STATE
)

# Fit postprocessor (following original)
eqodds = eqodds.fit(aif_val, aif_val_pred)

# Apply to test set
aif_fair_test_pred = eqodds.predict(aif_test_pred)
fair_predictions = aif_fair_test_pred.labels.ravel().astype(int)

# Save results (following original)
fair_results = test_df.copy()
fair_results['predicted'] = fair_predictions
#fair_results.to_csv('...', index=False)
fair_results.to_csv('fair_equalized_odds.csv')
print("Equalized Odds model completed")

# ------------------------------------------------------------------------
# MODEL 6: UNFAIR_EQUALISED_ODDS
# ------------------------------------------------------------------------

print("\nMODEL 6: UNFAIR_EQUALISED_ODDS")
print("=" * 40)
print("Description: Unfair wrt equalized odds")
print("Method: Always NO for women, always YES for men")

# Always NO for women, always YES for men
unfair_eq_odds_preds = np.ones_like(test_predictions)  # Start with all unfavorable (1=Default)
men_indices = (test_df[protected_attr] == 1)  # Men (Male = 1)
unfair_eq_odds_preds[men_indices] = 0  # Favorable (0=No Default) for men
# Women keep 1 (Default - unfavorable)

# Save results (following original)
unfair_eq_odds_results = test_df.copy()
unfair_eq_odds_results['predicted'] = unfair_eq_odds_preds
#unfair_eq_odds_results.to_csv('...', index=False)
unfair_eq_odds_results.to_csv('unfair_equalized_odds.csv', index=False)
print("Unfair Equalized Odds model completed")

# ------------------------------------------------------------------------
# MODEL 7: CONSTANT_NO
# ------------------------------------------------------------------------

print("\nMODEL 7: CONSTANT_NO")
print("=" * 40)
print("Description: Always predict NO")
print("Method: Predict 0 (No Default) for everyone")

# Always predict favorable (0=No Default) for everyone
constant_no_preds = np.zeros_like(test_predictions)

# Save results (following original)
constant_no_results = test_df.copy()
constant_no_results['predicted'] = constant_no_preds
#constant_no_results.to_csv('...', index=False)
constant_no_results.to_csv('constant_no.csv', index=False)
print("Constant NO model completed")

# ------------------------------------------------------------------------
# MODEL 8: CONSTANT_YES
# ------------------------------------------------------------------------

print("\n MODEL 8: CONSTANT_YES")
print("=" * 40)
print(" Description: Always predict YES")
print(" Method: Predict 1 (Default) for everyone")

# Always predict unfavorable (1=Default) for everyone
constant_yes_preds = np.ones_like(test_predictions)

# Save results (following original)
constant_yes_results = test_df.copy()
constant_yes_results['predicted'] = constant_yes_preds
#constant_yes_results.to_csv('...', index=False)
constant_yes_results.to_csv('constant_yes.csv', index=False)
print("Constant YES model completed")
