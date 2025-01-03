import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout

def create_simple_neural_network(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return history
