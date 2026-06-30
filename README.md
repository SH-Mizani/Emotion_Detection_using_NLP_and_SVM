# Emotion Detection using NLP and SVM

A Natural Language Processing (NLP) project that classifies the emotional category of English text using TF-IDF feature extraction and a Support Vector Machine (SVM) classifier.

## Overview

This project demonstrates a complete NLP pipeline, including text preprocessing, feature extraction, model training, evaluation, and prediction. The goal is to classify text into its corresponding emotion label.

## Features

- Text preprocessing
  - Remove HTML tags
  - Remove punctuation and numbers
  - Convert text to lowercase
  - Remove English stopwords

- Feature extraction using TF-IDF

- Emotion classification using Linear Support Vector Machine (LinearSVC)

- Model evaluation
  - Accuracy Score
  - Confusion Matrix

- Random prediction samples for manual inspection

---

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

---

## Machine Learning Pipeline

1. Load the dataset
2. Clean and preprocess text
3. Split the dataset into training and testing sets
4. Convert text into TF-IDF vectors
5. Train a Linear SVM classifier
6. Evaluate model performance
7. Visualize the confusion matrix
8. Predict emotions on unseen samples

---

## Project Structure

```
.
├── text_emotions.csv
├── emotion_detection.py
└── README.md
```

---

## Example Output

- Model Accuracy
- Confusion Matrix Visualization
- Random prediction examples

---

## Future Improvements

- Try Word2Vec or GloVe embeddings
- Experiment with transformer-based models (BERT)
- Hyperparameter tuning
- Improve text preprocessing
- Support more emotion categories

---

## Author

**Setareh Mizani**

Computer Engineering Student  