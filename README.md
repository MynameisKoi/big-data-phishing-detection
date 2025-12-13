# Big Data Phishing Detection System (CS570 Project)

## 📌 Project Overview
This project implements a scalable machine learning pipeline to detect phishing and spam emails. It leverages big data processing techniques to analyze large email corpora (SpamAssassin, Enron) and identifies malicious patterns using Natural Language Processing (NLP) and ensemble learning.

The system focuses on distinguishing legitimate "ham" emails from malicious "spam/phishing" attempts by analyzing both email metadata (time sent, sender info) and content (body text).

## 🚀 Key Features
* **Data Aggregation:** automated scripts to merge and standardize disparate datasets (SpamAssassin, Enron) into a unified format.
* **Exploratory Data Analysis (EDA):**
    * **Temporal Analysis:** Visualizes email traffic by hour-of-day to identify "botnet windows" vs. human behavior.
    * **Content Analysis:** Word clouds and frequency distributions for subject lines and body text.
* **Feature Engineering:**
    * TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
    * Hashing Vectorizer for memory-efficient text processing.
    * Extraction of metadata features like "Hour of Day" and email length.
* **Ensemble Modeling:** Implements a Voting Classifier combining **Naive Bayes**, **Random Forest**, and **Support Vector Machines (SVM)** to maximize detection accuracy.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `merge_dataset.ipynb` | **Data Pipeline:** Ingests raw CSVs, handles encoding errors, parses dates, and merges datasets into a clean master file. |
| `analysis.ipynb` | **EDA & Visualization:** Generates histograms for temporal analysis and word clouds to explore spam keywords. |
| `spam_classification.ipynb` | **Baseline Models:** Implementation of standard classifiers (Naive Bayes, Logistic Regression) to establish baseline metrics. |
| `ensemble.ipynb` | **Advanced Modeling:** Trains the Random Forest and Ensemble Voting classifiers, including hyperparameter tuning and evaluation. |
| `CS570 - PROJECT PAPER.pdf` | **Final Report:** Detailed documentation of methodology, architecture, and performance results. |
| `presentation.html` | **Project Slides:** A web-based presentation summarizing the project workflow and findings. |

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (RandomForest, MultinomialNB, VotingClassifier)
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **NLP:** NLTK (Stopwords, Tokenization)

## 📊 Methodology

### 1. Data Preprocessing
The pipeline handles the "messiness" of real-world email data by:
* Standardizing timestamps to UTC to handle multi-timezone datasets.
* Removing "garbage" dates (e.g., year 9999) and invalid headers.
* Tokenizing text and removing English stop words.

### 2. Model Architecture
We compared single models against ensemble methods:
* **Naive Bayes:** Used as a fast baseline for text classification.
* **Random Forest:** Captures non-linear relationships (e.g., time-of-day interactions).
* **Voting Classifier:** The final model aggregates predictions from multiple estimators to reduce False Positives.

### 3. Results
* **Accuracy:** The ensemble model achieved high accuracy in distinguishing spam from ham.
* **Behavioral Insight:** Phishing emails showed a distinct high-volume activity window between 9:00 AM and 2:00 PM UTC, contrasting with legitimate traffic patterns.

## 🔮 Future Improvements
* **Spark Integration:** Migrating the Pandas pipeline to **PySpark** for handling multi-terabyte datasets.
* **Deep Learning:** Implementing BERT or LSTM models for better semantic understanding of "spear-phishing" attempts.
* **Real-time Streaming:** Connecting the model to a Kafka stream for live email filtering.

## 🤝 Contributors
* *Khoi Duong* - CS570 Big Data Analytics

## 📜 License
This project is for educational purposes under the CS570 course requirements.