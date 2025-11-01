# ============================================================================
# HELPER FUNCTIONS FOR FAIRNESS EVALUATION
# ============================================================================
def safe_disparate_impact(metric):
    di = metric.disparate_impact()
    if np.isnan(di) or np.isinf(di):
        priv_rate = metric.selection_rate(privileged=True)
        unpriv_rate = metric.selection_rate(privileged=False)

        # Handle specific edge cases with different sentinel values
        if priv_rate == 0 and unpriv_rate == 0:
            return -999.0  # Both groups have 0 approvals (perfect equality in denial)
        elif abs(priv_rate - unpriv_rate) < 1e-10:
            return 1.0  # Perfect equality (non-zero rates)
        elif unpriv_rate == 0:
            return 999.0  # Only privileged group gets approvals (maximum bias)
        elif priv_rate == 0:
            return 0.001  # Only unprivileged group gets approvals (reverse bias)
    return min(max(di, 0.001), 999.0)

def calculate_all_metrics(y_true, y_pred, X_test, protected_attr, model_name):
    """
    Calculate comprehensive performance and fairness metrics
    """
    print(f"\nEVALUATING: {model_name}")
    print("-" * 40)

    # Basic performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases (only one class predicted)
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 0:  # All predicted as 0
                tn = len(y_true[y_true == 0])
                fp = 0
                fn = len(y_true[y_true == 1])
                tp = 0
            else:  # All predicted as 1
                tn = 0
                fp = len(y_true[y_true == 0])
                fn = 0
                tp = len(y_true[y_true == 1])
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

    print(f"   Accuracy: {accuracy:.3f} | Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"   TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

    # Create AIF360 datasets for fairness metrics
    test_df = X_test.copy()
    test_df['actual'] = y_true
    test_df['predicted'] = y_pred

    # For credit card: 0=No Default (favorable), 1=Default (unfavorable)
    # For HMDA: 1=Approved (favorable), 0=Denied (unfavorable)
    # We'll handle this in the specific dataset sections

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

print("Helper functions defined!")
