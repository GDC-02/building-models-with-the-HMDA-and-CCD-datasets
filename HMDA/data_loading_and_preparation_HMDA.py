# ============================================================================
#  HMDA DATA LOADING AND PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("HMDA LENDING FAIRNESS ANALYSIS")
print("=" * 80)

# HMDA data loading and preparation
hmda_file_path = '...'
print(f" Loading HMDA data from: {hmda_file_path}")

try:
    df_hmda = pd.read_csv(hmda_file_path)
    print(f"HMDA dataset loaded! Shape: {df_hmda.shape}")

    # Following your original HMDA preprocessing exactly
    print(" Filtering to approved and denied applications...")
    df_hmda_filtered = df_hmda[df_hmda['action_taken'].isin([1, 3])].copy()
    print(f"   After filtering: {len(df_hmda_filtered):,} applications")

    # Create binary target: 1=approved, 0=denied
    df_hmda_filtered['target_binary'] = df_hmda_filtered['action_taken'].map({1: 1, 3: 0})

    # Filter to Male and Female only
    df_hmda_filtered = df_hmda_filtered[df_hmda_filtered['applicant_sex'].isin([1, 2])].copy()
    print(f"   After gender filtering: {len(df_hmda_filtered):,} applications")

    # Create binary sex variable: 1=Male, 0=Female
    df_hmda_filtered['sex_binary'] = (df_hmda_filtered['applicant_sex'] == 1).astype(int)

    print(f" HMDA data prepared! Shape: {df_hmda_filtered.shape}")
    print(f" Target: target_binary (1=Approved, 0=Denied)")
    print(f" Protected: sex_binary (1=Male, 0=Female)")
    print(f"Target distribution: {df_hmda_filtered['target_binary'].value_counts().to_dict()}")
    print(f"Gender distribution: {df_hmda_filtered['sex_binary'].value_counts().to_dict()}")

except Exception as e:
    print(f" Error in HMDA data loading: {e}")
    print(" Please check HMDA file path and data format")
