from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_random_forest(X_train, y_train):
    """Train a Random Forest model with class weights."""
    unique_classes = np.unique(y_train)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    
    class_weights_dict = dict(zip(unique_classes, class_weights))
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weights_dict
    )
    
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    """Make predictions using the trained model."""
    return model.predict(X_test)

