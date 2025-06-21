# 📄 Fake News Detection using Machine Learning

A machine learning project that classifies news articles as fake or real using various algorithms including Naive Bayes, Logistic Regression, and Random Forest.

## 📊 Project Overview

This project implements a text classification system to detect fake news articles using natural language processing and machine learning techniques. The system preprocesses news text data and trains multiple models to achieve high accuracy in distinguishing between fake and real news.

## 🎯 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| **Random Forest** | **99.81%** | **1.00** | **1.00** | **1.00** |
| Logistic Regression | 98.53% | 0.99 | 0.99 | 0.99 |
| Naive Bayes | 92.17% | 0.92 | 0.92 | 0.92 |

## 🚀 Features

- **Text Preprocessing**: Removes non-alphabetic characters, converts to lowercase, and filters stop words
- **TF-IDF Vectorization**: Converts text to numerical features using Term Frequency-Inverse Document Frequency
- **Multiple Models**: Implements and compares three different machine learning algorithms
- **Performance Metrics**: Provides detailed accuracy, confusion matrix, and classification reports
- **High Accuracy**: Achieves up to 99.81% accuracy with Random Forest classifier

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Khushi-7git/FAKE-NEWS-DETECTION-.git
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

3. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## 📁 Dataset

The project uses two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

**Dataset Structure:**
- Articles are labeled as 0 (fake) or 1 (real)
- Text preprocessing removes noise and standardizes format
- Data is shuffled and split into training/testing sets (80/20)

## 🔧 Usage

1. Ensure your dataset files (`Fake.csv` and `True.csv`) are in the project directory
2. Run the Jupyter notebook:
```bash
jupyter notebook "FAKE NEWS DETECT.ipynb"
```

3. The notebook will:
   - Load and preprocess the data
   - Train three different models
   - Display performance metrics for each model

## 📈 Model Performance

### Random Forest (Best Performer)
- **Accuracy**: 99.81%
- **Confusion Matrix**: Only 17 misclassifications out of 8,980 test samples
- **Perfect precision and recall** for both classes

### Logistic Regression
- **Accuracy**: 98.53%
- **Well-balanced performance** across both classes
- **Fast training and prediction**

### Naive Bayes
- **Accuracy**: 92.17%
- **Good baseline performance**
- **Computationally efficient**

## 🔍 Text Preprocessing Pipeline

1. **Cleaning**: Remove non-alphabetic characters
2. **Normalization**: Convert to lowercase
3. **Tokenization**: Split into individual words
4. **Stop Word Removal**: Filter common English stop words
5. **Vectorization**: Apply TF-IDF with max 3000 features

## 🏗️ Model Architecture

- **Feature Extraction**: TF-IDF Vectorizer (max_features=3000, max_df=0.7)
- **Train/Test Split**: 80/20 ratio with random_state=42
- **Models Trained**:
  - Multinomial Naive Bayes
  - Logistic Regression (max_iter=1000)
  - Random Forest (n_estimators=100)

## 📊 Evaluation Metrics

For each model, the system provides:
- **Accuracy Score**: Overall correctness percentage
- **Confusion Matrix**: True vs predicted labels breakdown
- **Classification Report**: Precision, recall, and F1-score per class

## 🔮 Future Enhancements

- [ ] Add more sophisticated text preprocessing (stemming, lemmatization)
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add feature importance analysis
- [ ] Expand dataset with more diverse news sources
- [ ] Add cross-validation for more robust evaluation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact
AUTHOR - Khushi Kumari
---

⭐ If you found this project helpful, please give it a star!
