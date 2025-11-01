# ============================================================================
# CREDIT CARD DEFAULT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CREDIT CARD DEFAULT FAIRNESS ANALYSIS")
print("=" * 80)

# ----------------------------------------------------------------------------
# STEP 1: LOAD AND PREPARE CREDIT CARD DATA
# ----------------------------------------------------------------------------

print("\nSTEP 1: DATA LOADING AND PREPARATION")
print("-" * 50)

# Load credit card dataset
file_path = '...'
print(f"Loading from: {file_path}")

try:
    df_credit = pd.read_csv(file_path, delimiter=";", header=1)
    print(f"Loaded successfully! Shape: {df_credit.shape}")
except:
    print("Error loading file. Please check path and format.")
  
# Clean data
df_credit = df_credit.apply(pd.to_numeric, errors='coerce')
df_credit = df_credit.dropna()
df_credit = df_credit.rename(columns={'default payment next month': 'DEFAULT'})

# Prepare variables
label_col = 'DEFAULT'
protected_attr = 'SEX'

# Convert gender: 1=Male (privileged), 0=Female (unprivileged)
df_credit['SEX'] = df_credit['SEX'].map({1: 1, 2: 0})

print(f"Data prepared! Final shape: {df_credit.shape}")
print(f"Target: {label_col} (0=No Default, 1=Default)")
print(f"Protected: {protected_attr} (1=Male, 0=Female)")

# Show distributions
print(f"\nDistributions:")
print(f"   Default: {df_credit[label_col].value_counts().to_dict()}")
print(f"   Gender: {df_credit[protected_attr].value_counts().to_dict()}")

def encode_categorical_features(df):
    """
    Simple categorical encoding for credit card dataset
    """
    df_encoded = df.copy()

    print("Encoding categorical features...")

    # 1. EDUCATION: One-hot encode
    if 'EDUCATION' in df_encoded.columns:
        education_dummies = pd.get_dummies(df_encoded['EDUCATION'], prefix='EDUCATION', drop_first=True)
        df_encoded = pd.concat([df_encoded.drop('EDUCATION', axis=1), education_dummies], axis=1)
        print(f"✓ EDUCATION: One-hot encoded")

    # 2. MARRIAGE: One-hot encode
    if 'MARRIAGE' in df_encoded.columns:
        marriage_dummies = pd.get_dummies(df_encoded['MARRIAGE'], prefix='MARRIAGE', drop_first=True)
        df_encoded = pd.concat([df_encoded.drop('MARRIAGE', axis=1), marriage_dummies], axis=1)
        print(f"✓ MARRIAGE: One-hot encoded")

    # 3. Payment status: Keep as-is (they're already ordinal: -1, 0, 1, 2, etc.)
    payment_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    existing_payment_cols = [col for col in payment_cols if col in df_encoded.columns]
    if existing_payment_cols:
        print(f"✓ Payment status: Kept as ordinal ({len(existing_payment_cols)} columns)")

    print(f"Done! Features: {len(df.columns)} -> {len(df_encoded.columns)}")

    return df_encoded
# Apply encoding
df_credit = encode_categorical_features(df_credit)

