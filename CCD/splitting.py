# ----------------------------------------------------------------------------
# STEP 2: TRAIN/TEST SPLIT FOR CREDIT CARD
# ----------------------------------------------------------------------------

print("\nSTEP 2: TRAIN/TEST SPLIT")
print("-" * 50)

# Prepare features and target
X_credit = df_credit.drop(columns=[label_col])
y_credit = df_credit[label_col]

# Split into train/test
X_train, X_test_temp, y_train, y_test_temp = train_test_split(
    X_credit, y_credit, test_size=0.5, random_state=RANDOM_STATE, stratify=y_credit
)

# Split temp into validation and test
combined_strata = [f"{label}_{gender}" for label, gender in
                  zip(y_test_temp, X_test_temp[protected_attr])]

try:
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_temp, y_test_temp, test_size=0.5, random_state=RANDOM_STATE,
        stratify=combined_strata
    )
    print("Stratified split successful")
except ValueError:
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_temp, y_test_temp, test_size=0.5, random_state=RANDOM_STATE,
        stratify=y_test_temp
    )
    print("Simple stratified split used")

print(f"Sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
