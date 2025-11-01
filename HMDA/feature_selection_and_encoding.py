# ========================================================================
# HMDA FEATURE SELECTION (following original structure)
# ========================================================================

print("\nHMDA FEATURE SELECTION")
print("-" * 50)

# Select key features for modeling
numerical_features_hmda = []
if 'loan_amount' in df_hmda_filtered.columns:
    numerical_features_hmda.append('loan_amount')
if 'income' in df_hmda_filtered.columns:
    numerical_features_hmda.append('income')

# Categorical features
categorical_features_hmda = []
for col in ['loan_purpose', 'loan_type', 'preapproval', 'occupancy_type', 'construction_method', 'lien_status', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 'applicant_age', 'applicant_ethnicity-1', 'business_or_commercial_purpose', 'balloon_payment', 'interest_only_payment', 'negative_amortization']:
    if col in df_hmda_filtered.columns:
        categorical_features_hmda.append(col)

print(f" Numerical features: {numerical_features_hmda}")
print(f" Categorical features: {categorical_features_hmda}")

# Handle missing values and encode categoricals
print("Handling missing values and encoding...")

# Fill missing numerical values with median
for col in numerical_features_hmda:
    if df_hmda_filtered[col].isnull().sum() > 0:
        median_val = df_hmda_filtered[col].median()
        df_hmda_filtered[col].fillna(median_val, inplace=True)
        print(f"   {col}: filled missing with median {median_val}")

# Fill missing categorical values and encode
label_encoders_hmda = {}
encoded_features_hmda = []
encoding_mappings_hmda = {}  # New dictionary to store mappings

for col in categorical_features_hmda:
    if col in df_hmda_filtered.columns:
        # Fill missing values
        if df_hmda_filtered[col].isnull().sum() > 0:
            mode_val = df_hmda_filtered[col].mode()
            if len(mode_val) > 0:
                df_hmda_filtered[col].fillna(mode_val[0], inplace=True)
            else:
                df_hmda_filtered[col].fillna('Unknown', inplace=True)

        # Encode categorical
        le = LabelEncoder()
        encoded_col = f"{col}_encoded"
        df_hmda_filtered[encoded_col] = le.fit_transform(df_hmda_filtered[col].astype(str))
        label_encoders_hmda[col] = le
        encoded_features_hmda.append(encoded_col)

        # Mapping from original values to encoded values
        unique_values = df_hmda_filtered[col].unique()
        encoded_values = le.transform(unique_values.astype(str))
        mapping = dict(zip(unique_values, encoded_values))
        encoding_mappings_hmda[col] = mapping

# Final feature list
final_features_hmda = numerical_features_hmda + encoded_features_hmda
print(f"Final HMDA features ({len(final_features_hmda)}): {final_features_hmda}")

# ========================================================================
# Visualizing ENCODING
# ========================================================================

def show_encoding_summary():
    print("SUMMARY OF ALL ENCODINGS")

    for col, mapping in encoding_mappings_hmda.items():
        print(f"\n{col.upper()}:")
        print("-" * len(col))
        for original, encoded in sorted(mapping.items(), key=lambda x: x[1]):
            print(f"  {encoded:2d}: {original}")
show_encoding_summary()
'''
def get_original_value(column, encoded_value):
    """Get ogiginal value"""
    if column in label_encoders_hmda:
        try:
            return label_encoders_hmda[column].inverse_transform([encoded_value])[0]
        except ValueError:
            return f"Encoded value {encoded_value} not found for column {column}"
    else:
        return f"Column {column} not found in encoders"

def get_encoded_value(column, original_value):
    """Get encode value"""
    if column in label_encoders_hmda:
        try:
            return label_encoders_hmda[column].transform([str(original_value)])[0]
        except ValueError:
            return f"Original value '{original_value}' not found for column {column}"
    else:
        return f"Column {column} not found in encoders"

def show_encoding_for_column(column):
    """Encoding for specific column"""
    if column in encoding_mappings_hmda:
        print(f"\nEncoding for {column}:")
        print("-" * (len(column) + 12))
        mapping = encoding_mappings_hmda[column]
        for original, encoded in sorted(mapping.items(), key=lambda x: x[1]):
            count = (df_hmda_filtered[column] == original).sum()
            print(f"  {encoded:2d}: {original} (count: {count})")
    else:
        print(f"Column {column} not found in encodings")

# HOW TO USE ABOVE FUNCTIONS

if 'loan_purpose' in encoding_mappings_hmda:
    print("\nExample 1: Encoding for loan_purpose")
    show_encoding_for_column('loan_purpose')

if 'loan_purpose' in label_encoders_hmda:
    print(f"\nExample 2: Encoded value 0 in loan_purpose = '{get_original_value('loan_purpose', 0)}'")

if 'loan_purpose' in label_encoders_hmda:
    first_category = list(encoding_mappings_hmda['loan_purpose'].keys())[0]
    encoded_val = get_encoded_value('loan_purpose', first_category)
    print(f"Example 3: Original value '{first_category}' in loan_purpose = {encoded_val}")
'''
# ========================================================================
#CREATING DF FOR CONSULTING MAPPINGS
# ========================================================================

def create_encoding_dataframe():
    all_mappings = []

    for col, mapping in encoding_mappings_hmda.items():
        for original, encoded in mapping.items():
            count = (df_hmda_filtered[col] == original).sum()
            all_mappings.append({
                'column': col,
                'original_value': original,
                'encoded_value': encoded,
                'count': count
            })

    return pd.DataFrame(all_mappings)

encoding_df = create_encoding_dataframe()
print("\n" + "="*60)
print("ENCODING DATAFRAME (first 10 rows)")
print("="*60)
print(encoding_df.head(10))

encoding_df.to_csv('hmda_encoding_mappings.csv', index=False)
print("\nEncoding mappings saved to 'hmda_encoding_mappings.csv'")
