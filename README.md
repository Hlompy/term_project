# Product Recommendation System
A machine learning-based product recommendation system that suggests complementary products based on user selections. This project works with dataset of Amazon Sales(https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset), where buying a phone triggers recommendations for accessories like cases and chargers.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview
This project implements a recommendation system that suggests related products based on user purchases. For example, when a user buys a phone, the system recommends accessories like phone cases, chargers, or headphones.

The project is divided into the following major components:
1. **Data Preprocessing**: Cleaning and preparing the data for modeling.
2. **Traditional Machine Learning**: Implementation using Random Forest.
3. **Neural Network**: Implementation using TensorFlow and Keras.
4. **Evaluation**: Comparing performance metrics between the two approaches.

---

## Features
- Predicts complementary products based on existing purchases.
- Two modeling approaches: Random Forest and Neural Network.
- Automatically preprocesses data, including handling missing values and encoding categories.
- Performance comparison with metrics such as Accuracy, Precision, Recall, and F1 Score.

---

## Technologies Used
- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Matplotlib (for visualization)

---

## Project Structure
```
project.folder
├── data
├── models
│   ├── traditional_ml.py
│   └── neural_networks.py
├── utils
│   ├── data_preprocessing.py
│   └── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```


---

## Setup Instructions
1. Clone the repository:
```bash
   git clone https://github.com/yourusername/product-recommendation-system.git
   cd product-recommendation-system
```
2. Create a virtual environment and activate it:
```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Place the dataset (amazon.csv) in the data/ directory.


## Usage
1. Run the main script to preprocess the data, train the models, and evaluate their performance:
```bash
python main.py
```

2.The output will include:

    Metrics for both Random Forest and Neural Network models.
    Details about data preprocessing and class distribution.


---


## Model Training and Evaluation
Random Forest

- Used as a baseline model.
- Handles class imbalance with `class_weight` parameter.

### Neural Network
- Consists of fully connected layers implemented using TensorFlow/Keras.
- Optimized with Adam optimizer and Binary Cross-Entropy loss.

### Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## Future Enhancements
- Add collaborative filtering methods for recommendation.
- Implement deep learning architectures like Recurrent Neural Networks (RNN) for sequence-based predictions.
- Deploy the system as a web service using Flask or FastAPI.
- Visualize recommendations on a user-friendly interface.

---

## Acknowledgments
- Dataset: [Amazon Product Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- Inspired by real-world recommendation systems and Amazon.

---

## Contact
- **GitHub**: [Hlompy](https://github.com/Hlompy)


