# 🧠 Comment Category Prediction

> Classifying social media / YouTube comments into 4 categories using NLP + LightGBM

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io)
[![TF-IDF](https://img.shields.io/badge/NLP-TF--IDF-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Problem Statement

Given a social media comment along with its metadata, predict which of **4 categories** it belongs to:

| Label | Category |
|-------|----------|
| 0 | Normal |
| 1 | Identity Hate |
| 2 | Offensive |
| 3 | Violent / Toxic |

---

## 📊 Dataset

- **Train set:** 198,000 samples × 15 features
- **Test set:** 102,000 samples × 14 features

### Features include:
- `comment` — the raw text of the comment
- `emoticon_1/2/3` — emoticons used in the comment
- `upvote`, `downvote` — engagement metrics
- `race`, `religion`, `gender`, `disability` — identity-related metadata
- `created_date`, `post_id` — temporal and post metadata

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas`, `numpy` | Data loading & manipulation |
| `scikit-learn` | TF-IDF, preprocessing, metrics |
| `LightGBM` | Classification model |
| `matplotlib`, `seaborn` | EDA & visualization |
| `scipy` | Sparse matrix operations |

---

## 🚀 Workflow

```
1. Load Data
      ↓
2. Exploratory Data Analysis (EDA)
      ↓
3. Advanced Text Cleaning
   - Lowercase conversion
   - Remove URLs, emails, HTML tags
   - Remove special characters & numbers
      ↓
4. Feature Engineering
   - TF-IDF (Word-level): 50,000 features, 1–2 grams
   - TF-IDF (Char-level): 30,000 features, 3–5 grams
   - Ordinal encoding for categorical features
   - MinMaxScaler for numerical features
   - Sparse matrix stacking (hstack)
      ↓
5. Model Training — LightGBM
      ↓
6. Retrain on Full Data
      ↓
7. Generate Submission
```

---

## 🧹 Text Cleaning Pipeline

```python
def advanced_text_cleaning(text):
    text = str(text).lower()                            # Lowercase
    text = re.sub(r'http\S+|www\S+|...', '', text)     # Remove URLs & emails
    text = re.sub(r'<[^>]+>', '', text)                 # Remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', ' ', text)           # Special characters
    text = re.sub(r'\b\d+\b', '', text)                 # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()            # Clean whitespace
    return text
```

---

## 📈 Model: LightGBM

**Why LightGBM?**
- Handles large sparse matrices efficiently (perfect for TF-IDF)
- Fast training on 198K samples
- Excellent performance on imbalanced multi-class problems

**Key Parameters:**
```python
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'num_leaves': 127,
    'boosting_type': 'gbdt'
}
```

---

## 📁 Project Structure

```
comment-category-prediction/
│
├── comment_category_prediction.ipynb   # Main Jupyter Notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── LICENSE                             # MIT License
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/anjali-kokare/comment-category-prediction.git
cd comment-category-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your dataset
# Place train.csv and test.csv in the project root

# 4. Run the notebook
jupyter notebook comment_category_prediction.ipynb
```

---

## 📦 Requirements

See `requirements.txt` for full list. Key packages:
- `lightgbm`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

---

## 🏆 Evaluation Metric

**Macro F1-Score** — equal weight given to all 4 classes, handles class imbalance fairly.

---

## 👩‍💻 Author

**Anjali Kokare**
- 🎓 BS Data Science & AI — IIT Madras
- 🔗 [LinkedIn](https://www.linkedin.com/in/anjali-kokare-645488382)
- 📧 Open to collaborations & opportunities!

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
