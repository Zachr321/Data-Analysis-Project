# Toxic Comment Detection — Bag of Words

Binary classifier that detects toxic comments using a Bag-of-Words approach on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Zachr321/Data-Analysis-Project.git
cd Data-Analysis-Project
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the notebook

Open `Notebook.ipynb` in Jupyter or VS Code and **Run All** cells.

### Run the Python scripts

Select the Python 3.13 interpreter in VS Code, then install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the preprocessing and TF-IDF pipeline from the repository root:

```bash
python preprocess.py
python train_tfidf.py --data out --preds preds
```

To train the optional BERT model instead:

```bash
python train_bert.py --data out --preds preds
```

The BERT model downloads its weights on first use and is much faster with a supported GPU.

## Project Structure

```
Data-Analysis-Project/
├── Notebook.ipynb              # Main analysis notebook
├── requirements.txt            # Python dependencies
├── train.csv/train.csv         # Training data (159,571 comments)
├── test.csv/test.csv           # Test data
├── test_labels.csv/            # Test labels
├── sample_submission.csv/      # Kaggle submission format
├── all reports/                # Course reports (PDF)
└── README.md
```

## Pipeline

1. **Data Loading** — Read CSVs with pandas
2. **Text Preprocessing** — Clean HTML, URLs, emails, IPs → lowercase
3. **Feature Extraction** — Bag of Words (`CountVectorizer`)
4. **Modeling** — Multinomial Naive Bayes + Logistic Regression
5. **Evaluation** — Classification report, confusion matrix, model comparison

## Results (Validation Set)

| Model               | Accuracy | F1 (Toxic) |
|---------------------|----------|------------|
| Naive Bayes (BoW)   | 0.95     | 0.70       |
| Logistic Regression  | TBD      | TBD        |
