# ========================================================================
# HMDA TRAIN/TEST SPLIT
# ========================================================================

print("\nHMDA TRAIN/TEST SPLIT")
print("-" * 50)

# Define HMDA variables
hmda_label_col = 'target_binary'
hmda_protected_attr = 'sex_binary'

# Prepare HMDA features and target
X_hmda = df_hmda_filtered[final_features_hmda + [hmda_protected_attr]]
y_hmda = df_hmda_filtered[hmda_label_col]

print(f"HMDA X shape: {X_hmda.shape}")
print(f"HMDA y shape: {y_hmda.shape}")

# Train/test split
X_train_hmda, X_test_temp_hmda, y_train_hmda, y_test_temp_hmda = train_test_split(
    X_hmda, y_hmda, test_size=0.5, random_state=RANDOM_STATE, stratify=y_hmda
)

# Validation/test split
combined_strata_hmda = [f"{label}_{gender}" for label, gender in
                       zip(y_test_temp_hmda, X_test_temp_hmda[hmda_protected_attr])]

try:
    X_val_hmda, X_test_hmda, y_val_hmda, y_test_hmda = train_test_split(
        X_test_temp_hmda, y_test_temp_hmda, test_size=0.5, random_state=RANDOM_STATE,
        stratify=combined_strata_hmda
    )
    print("HMDA stratified split successful")
except ValueError:
    X_val_hmda, X_test_hmda, y_val_hmda, y_test_hmda = train_test_split(
        X_test_temp_hmda, y_test_temp_hmda, test_size=0.5, random_state=RANDOM_STATE,
        stratify=y_test_temp_hmda
    )
    print("HMDA simple stratified split used")

print(f"HMDA Sizes - Train: {len(X_train_hmda)}, Val: {len(X_val_hmda)}, Test: {len(X_test_hmda)}")

# Create test dataset for storing results (following original)
test_df_hmda = X_test_hmda.copy()
test_df_hmda['actual'] = y_test_hmda

# Scale HMDA features
scaler_hmda = StandardScaler()
X_train_hmda_scaled = scaler_hmda.fit_transform(X_train_hmda)
X_val_hmda_scaled = scaler_hmda.transform(X_val_hmda)
X_test_hmda_scaled = scaler_hmda.transform(X_test_hmda)

print("HMDA features scaled")
