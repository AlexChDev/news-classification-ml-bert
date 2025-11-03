# News Category Classifier — Classical ML, Neural Networks & BERT (Finance/Health/Sports/Tech)

## 🧩 Overview

Repository for **classifying news articles** into four categories: **Finance, Health, Sports, Technology**.
The project compares different approaches on the **same task**: classical models (TF-IDF/Bag-of-Words), **Keras neural networks**, and **Transformers** (mBERT). The notebook was developed/run in **Google Colab** and saved as a `.ipynb` with outputs included.

---

## 📦 Data

* **File:** `news_articles_dataset_mod_v1.csv`
* **Columns used:** `Text` (text) and `Category` (label).
* **Classes:** `Finance`, `Health`, `Sports`, `Technology`.

### Data split

The notebook contains **two split schemes**, consistent with the sections that use them:

* **Classical models & Keras** → `80%` train / `10%` validation / `10%` test (random_state=42).
* **BERT (HuggingFace)** → `56%` train / `14%` validation / `30%` test (obtained from a 70/30 split with 20% of the training set used as validation; random_state=13).

> Note: the different splits reflect the practice used in the BERT section with `datasets`/`Trainer`.

---

## 🔧 Models & Configurations

### 1) Classical baselines

* **TF-IDF + Logistic Regression** (scikit-learn).
* **CountVectorizer + Multinomial Naive Bayes**.
* **CountVectorizer + Logistic Regression**.
* **CountVectorizer + RandomForestClassifier**.
  For each model: **classification_report**, **confusion matrix**, and a final comparison of metrics (Accuracy, Precision, Recall, F1) in a **2×2 dashboard**.

### 2) Neural network (Keras)

* **Dense NN** on **TF-IDF** features obtained with `TextVectorization(output_mode='tf-idf')`.
* **EarlyStopping** and training/validation curves shown in the notebook.

### 3) RNN with word embeddings

* **LSTM** with **pre-trained** `word2vec-google-news-300` weights (via `gensim.downloader`), tokenization with `Tokenizer`, sequence padding, and a **non-trainable** (frozen) embedding matrix.

### 4) Transformer (Hugging Face)

* **Model:** `bert-base-multilingual-uncased` (mBERT).
* **Tokenization:** `AutoTokenizer`, padding/truncation to a fixed length.
* **Training:** `Trainer` with `TrainingArguments(evaluation_strategy="epoch")`.
* **Model saving:** directory with the `"_NEWS"` suffix (e.g., `bert-base-multilingual-uncased_NEWS`).
* **Evaluation:** use of `pipeline(task="text-classification")` on the test set and `classification_report`.

---

## 📊 Results (summary)

> The metrics below are those **printed in the notebook** on the respective test sets.

| Model                                       | Accuracy | Weighted F1 | Test set / Notes                                            |
|--------------------------------------------|:--------:|:-----------:|-------------------------------------------------------------|
| TF-IDF + Logistic Regression               | **0.80** |    ≈ 0.79   | Test support: **415**; strong on `Finance`/`Technology`, harder on `Health`. |
| CountVectorizer + Multinomial Naive Bayes  |  0.77    |    ≈ 0.76   | Test support: **415**.                                      |
| Dense NN (TF-IDF/Keras)                    |  0.70    |    ≈ 0.67   | Test support: **415**.                                      |
| LSTM + word2vec-google-news-300            |  0.79    |    ≈ 0.79   | Test support: **415**.                                      |
| mBERT fine-tuned                           | **0.82** |    ≈ 0.82   | Test set ≈ **30%** of data (≈ **1,244** samples).           |

> Quick observation: the **TF-IDF + LR** baseline is very solid; the **LSTM** comes close and **mBERT** is the best in the reported set-up. The models struggle with the **Health** class more than the others.

---

## ▶️ How to run

1. Open `comparative-news-classifier.ipynb` in **Google Colab**.
2. Run the cells in order (the necessary `pip install`s are included: `transformers`, `datasets`, `evaluate`, `accelerate`).
3. Make sure the file `news_articles_dataset_mod_v1.csv` is available in Colab’s working path (or mount Drive).

### Main requirements

`pandas, scikit-learn, matplotlib, seaborn, tensorflow/keras, gensim, transformers, datasets, evaluate, accelerate`

---

## 📁 Notebook structure (sections)

1. **TF/IDF** (vectorization and LR)
2. **Naive Bayes** (CountVectorizer)
3. **Logistic Regression** (CountVectorizer)
4. **RandomForest** (CountVectorizer)
5. **Neural network** (Keras Dense on TF-IDF)
6. **Neural network with Word embeddings** (LSTM + word2vec)
7. **BERT** (tokenization, training with `Trainer`, saving, evaluation)

---

## 📌 Notes

* The **class distribution** is skewed toward `Finance`/`Technology` (see details in the confusion matrices).
* The **splits differ** between the classical/Keras section and the BERT one (see above).
* Metrics may vary slightly across runs due to residual randomness.

---
