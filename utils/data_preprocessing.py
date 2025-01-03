import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

import pandas as pd

def preprocess_data(df):
    """Data preprocessing for the model."""
    print(f"Initial number of entries: {len(df)}")

    df = df.dropna()
    print(f"After removing gaps: {len(df)}")

    try:
        df.loc[:, 'discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
        print("The discounted_price conversion is complete.")
    except Exception as e:
        print(f"Error converting discounted_price: {e}")
    # Преобразование rating в числовой формат и заполнение пропусков
    try:
        df.loc[:, 'rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.loc[:, 'rating'] = df['rating'].fillna(df['rating'].mean())
        print("The rating conversion is complete.")
    except Exception as e:
        print(f"Error converting rating: {e}")

    # Диагностика уникальных значений product_id
    print("Unique values ​​in product_id before removing rare classes:")
    print(df['product_id'].value_counts())

    # Удаление редких классов (порог изменен на 2 для сохранения данных)
    if 'product_id' in df.columns:
        class_counts = df['product_id'].value_counts()
        rare_classes = class_counts[class_counts < 2].index  # Порог снижен до 2
        df = df[~df['product_id'].isin(rare_classes)]
        print(f"After removing rare classes: {len(df)}")
    else:
        print("The 'product_id' column is missing from the data.")

    # Кодирование категорий
    try:
        if 'category' in df.columns:
            df.loc[:, 'category_encoded'] = pd.factorize(df['category'])[0]
            print("Categories coding is complete.")
        else:
            print("The 'category' column is missing from the data.")
    except Exception as e:
        print(f"Error while encoding categories: {e}")

    # Итоговая диагностика
    print(f"Processed data: {len(df)} records")
    return df



def split_data(df):
    """Split dataset into training and testing sets."""
    # Выбираем признаки и целевую переменную
    X = df[['category_encoded', 'discounted_price', 'rating']]
    y = pd.factorize(df['product_id'])[0]

    if X.empty or len(y) == 0:
        raise ValueError("The characteristics or target variable are empty. Check data processing.")
    
    # Преобразуем y в числовой формат
    y = pd.factorize(y)[0]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_neural_network(model, X_train, y_train):
    """Train the neural network."""
    # Проверяем, что X_train и y_train - это числовые данные
    assert X_train.dtypes.isin([float, int]).all(), "X_train should only store numeric data"
    assert pd.api.types.is_numeric_dtype(y_train), "y_train must be numeric"
    
    # Преобразуем y_train в формат float32
    y_train = y_train.astype('float32')
    
    # Обучение модели
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return history


