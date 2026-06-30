import pandas as pd
import re
import nltk
import random
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# Data directory and read from CSV file
# The dataset contains two columns: the text samples and their corresponding emotion labels.
df = r'C:\Users\Asus\Desktop\EDU\project-nlp\text_emotions.csv'
data = pd.read_csv(df, names=["text", "label"], sep=",")

# Download stopwords from nltk and initialize the English stopword set
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing to remove punctuation marks and non-alphabetic characters, then converts text to lowercase
def preprocessing(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing to all text samples
data['cleaned_text'] = data['text'].apply(preprocessing)

# Train model
# Split the dataset into training and testing sets
x = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Word embeding using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
# Using SVM classifier to train features
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Visualize the confusion matrix using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Display 10 random samples from the dataset to manually inspect model predictions
print("\n10 random samples: ")
indices = list(range(len(X_test)))
random.shuffle(indices)
sample_indices = indices[:10]
for index in sample_indices:
    original_text = data.loc[index, "text"]
    cleaned_text = data.loc[index, "cleaned_text"]
    true_label = data.loc[index, "label"]
    vec = vectorizer.transform([cleaned_text])
    pred_label = model.predict(vec)[0]
    print("sentence: ", original_text)
    print("true label: ", true_label)
    print("prediction: ", pred_label)
    print("correct? ", "YES" if true_label == pred_label else "NO")
    print(" - " * 40)
