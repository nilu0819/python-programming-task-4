# python-programming-task-4
# ML model to detect a spam email 

# 📦 Install and import required packages
!pip install pandas numpy scikit-learn nltk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# 📥 Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# ✅ Preprocessing
df["label"] = df["label"].map({"ham": 0, "spam": 1})
stop_words = stopwords.words('english')

# 🧹 Text Vectorization
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# 🔁 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🧠 Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 📈 Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#output
Accuracy: 0.9784688995215312 precision recall f1-score support 0 0.99 0.98 0.99 1448 1 0.89 0.96 0.92 224 accuracy 0.98 1672 macro avg 0.94 0.97 0.95 1672 weighted avg 0.98 0.98 0.98 1672

<img width="1386" height="214" alt="Screenshot 2025-07-23 210535" src="https://github.com/user-attachments/assets/fea9e43b-1e50-42d7-a485-43948837cff3" />
