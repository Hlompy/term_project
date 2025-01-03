from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score, f1_score)

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy score."""
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    """Calculate precision score."""
    return precision_score(y_true, y_pred, average='weighted', zero_division=0)

def calculate_recall(y_true, y_pred):
    """Calculate recall score."""
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)

def calculate_f1(y_true, y_pred):
    """Calculate F1 score."""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def print_metrics(y_true, y_pred):
    """Print all metrics."""
    print(f"Accuracy: {calculate_accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {calculate_precision(y_true, y_pred):.4f}")
    print(f"Recall: {calculate_recall(y_true, y_pred):.4f}")
    print(f"F1 Score: {calculate_f1(y_true, y_pred):.4f}")
