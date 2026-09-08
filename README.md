# Sentiment Analysis Pipeline

This repository contains an end-to-end Natural Language Processing (NLP) pipeline for Sentiment Analysis. The notebook processes raw text data, cleans it, extracts features, and trains multiple machine learning models to classify emotions accurately. 

## 🛠️ Libraries & Dependencies
* **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`
* **Text Processing:** `string`, `nltk` (stopwords, punkt)
* **Machine Learning:** `scikit-learn` (Feature Extraction, Model Selection, Metrics)
* **Advanced Ensembles:** `lightgbm`

## 🗂️ Dataset
* **Source:** `train.txt`
* **Format:** Semicolon-separated values (`sep=';'`)
* **Columns:** `text` (the text snippet) and `emotion` (the target label).
* The categorical string labels in the `emotion` column are mapped to numerical representations for modeling.

## ⚙️ Data Preprocessing Pipeline
To prepare the raw text for the machine learning models, the following cleaning steps are applied sequentially:
1. **Lowercasing:** Converts all text to lowercase to ensure uniformity.
2. **Punctuation Removal:** Strips all punctuation marks using the `string.punctuation` module.
3. **Number Removal:** Eliminates all numerical digits from the text.
4. **Emoji & Non-ASCII Removal:** Filters out emojis and special characters by retaining only ASCII characters.
5. **Stopword Removal:** Removes common, uninformative English words (like 'the', 'is', 'in') using `NLTK`'s standard stopword corpus.

## 🧠 Feature Engineering
The text data is converted into numerical vectors using two primary techniques:
* **Bag-of-Words (CountVectorizer):** Calculates word frequencies.
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Evaluates how relevant a word is to a document in a collection.
* **Advanced TF-IDF:** Incorporates n-grams `(1, 2)` (Unigrams and Bigrams) and `sublinear_tf=True` scaling to capture short phrase contexts and boost accuracy.

## 🤖 Models & Performance
The dataset is split into an 80/20 Train-Test configuration. Several models were trained and evaluated:

| Model | Feature Extraction | Accuracy |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | Bag of Words (CountVectorizer) | ~76.81% |
| **Multinomial Naive Bayes** | standard TF-IDF | ~66.09% |
| **LightGBM Classifier** | standard TF-IDF | ~86.25% |
| **Logistic Regression** | standard TF-IDF | ~86.28% |
| **LinearSVC (SVM)** | standard TF-IDF | ~89.18% |
| **LinearSVC (Tuned)** | **TF-IDF (Bigrams + Sublinear)** | **~90.37%** 🏆 |

## 🚀 Conclusion
The **Linear Support Vector Machine (LinearSVC)** combined with **TF-IDF Bigrams** proved to be the most effective model, achieving over 90% accuracy in correctly classifying the text emotions.
