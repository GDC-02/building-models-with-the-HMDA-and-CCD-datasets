# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Features scaled with StandardScaler")

# ----------------------------------------------------------------------------
# STEP 3: CREATE AIF360 DATASETS FOR CREDIT CARD
# ----------------------------------------------------------------------------

# Create test dataset for storing results (following original)
test_df = X_test.copy()
test_df['actual'] = y_test
print("\nSTEP 3: CREATE AIF360 DATASETS")
print("-" * 50)

# Create AIF360 datasets
# For credit card: 0=No Default (favorable), 1=Default (unfavorable)
aif_train = BinaryLabelDataset(
    favorable_label=0, unfavorable_label=1,
    df=X_train.assign(**{label_col: y_train.values}),
    label_names=[label_col],
    protected_attribute_names=[protected_attr]
)

aif_val = BinaryLabelDataset(
    favorable_label=0, unfavorable_label=1,
    df=X_val.assign(**{label_col: y_val.values}),
    label_names=[label_col],
    protected_attribute_names=[protected_attr]
)

aif_test = BinaryLabelDataset(
    favorable_label=0, unfavorable_label=1,
    df=X_test.assign(**{label_col: y_test.values}),
    label_names=[label_col],
    protected_attribute_names=[protected_attr]
)

print("AIF360 datasets created")
