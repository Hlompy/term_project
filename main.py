import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_preprocessing import load_data, preprocess_data, split_data
from models.traditional_ml import train_random_forest, make_predictions
from models.neural_networks import create_simple_neural_network, train_neural_network
from utils.evaluation import print_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Device used:", tf.config.list_physical_devices('CPU'))

def main():
    print("Loading Data...")
    data = load_data('data/amazon.csv')
    print(f"Data loaded. Number of records: {len(data)}")

    print("Data preprocessing...")
    processed_data = preprocess_data(data)
    print(f"The data has been processed. Number of records: {len(processed_data)}")

    if processed_data.empty:
        raise ValueError("The processed data is empty. Check the pre-processing steps.")

    print("Dividing data into training and test sets...")
    X_train, X_test, y_train, y_test = split_data(processed_data)
    print(f"The data is separated. Training records: {len(X_train)}, Test records: {len(X_test)}")


    print("\nTraining the Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    rf_predictions = make_predictions(rf_model, X_test)
    print("Random Forest trained.")
    print("\nMetrics for Random Forest:")
    print_metrics(y_test, rf_predictions)

    print("\nTraining a neural network model...")
    nn_model = create_simple_neural_network(X_train.shape[1])
    train_neural_network(nn_model, X_train, y_train)
    print("Neural network trained.")

    print("\nUnique classes in y_train:")
    print(np.unique(y_train))

    nn_predictions = nn_model.predict(X_test)
    nn_predictions = (nn_predictions > 0.5).astype(int)
    print("\nMetrics for a neural network:")
    print_metrics(y_test, nn_predictions)

if __name__ == "__main__":
    main()