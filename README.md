# Email Spam Detection Using Machine Learning Algorithms

A Django web application that classifies email messages as **spam** or **ham**, comparing a plain Random Forest classifier against a Random Forest trained on features selected by a **Genetic Algorithm**.

## Overview

Traditional spam filters train on every feature a TF-IDF vectoriser produces, which is slow and noisy. This project demonstrates that evolutionary feature selection can shrink the feature space dramatically while keeping — or improving — classification quality.

The application ships an admin console where you can upload and browse the dataset, train both variants, view side-by-side accuracy/precision/recall/F-score tables, and classify an arbitrary message on demand.

## Features

- **Admin authentication** — simple login gate before the training console.
- **Dataset upload & preview** — render the spam/ham corpus as an HTML table.
- **Baseline training** — TF-IDF features + Random Forest, with the fitted model cached to disk.
- **Genetic-algorithm training** — `GeneticSelectionCV` evolves a population of 50 over 10 generations to select the 5 most discriminative features, then fits a Random Forest on that subset.
- **Metrics comparison** — accuracy, precision, recall and F1-score reported per approach.
- **Live spam detection** — paste a message and get an immediate spam/ham verdict.
- **Text preprocessing** — punctuation stripping, stop-word removal and WordNet lemmatisation shared across training and inference.

## Tech stack

| Layer | Technology |
|---|---|
| Web framework | Django |
| Language | Python 3.7 |
| ML | scikit-learn (Random Forest, TF-IDF), `sklearn-genetic` |
| NLP | NLTK |
| Data | pandas, NumPy |
| Storage | SQLite (`db.sqlite3`), PyMySQL driver available |
| Frontend | Django templates, CSS |

## Project structure

```
.
├── manage.py                   # Django entry point
├── run.bat                     # python manage.py runserver
├── db.sqlite3
├── Dataset/
│   ├── spam_ham_dataset.csv    # Labelled training corpus
│   └── testMessages.txt        # Sample messages for prediction
├── model/
│   ├── X.txt.npy, Y.txt.npy    # Cached feature matrix and labels
│   ├── tfidf.txt               # Pickled TF-IDF vectoriser
│   └── rf.txt                  # Pickled Random Forest model
├── SpamDetection/              # Project settings, URLs, WSGI
└── SpamDetectionApp/
    ├── views.py                # All request handlers and the ML pipeline
    ├── urls.py
    ├── static/                 # CSS and images
    └── templates/              # Login, upload, train, detect, results pages
```

## Getting started

### Prerequisites

- Python 3.7

### Installation

```bash
git clone https://github.com/srikanth-sri756/10.Email-Spam-Detection-Using-Machine-Learning-Algorithms.git
cd 10.Email-Spam-Detection-Using-Machine-Learning-Algorithms

pip install django numpy pandas scikit-learn nltk sklearn-genetic pymysql
```

Fetch the NLTK data used for cleaning:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running

```bash
python manage.py runserver
```

or double-click `run.bat` on Windows, then open <http://127.0.0.1:8000/>.

## Usage

1. Open the home page and go to **Login**. Sign in with the built-in admin credentials (`admin` / `admin`).
2. **Upload Dataset** — load `Dataset/spam_ham_dataset.csv` and preview the labelled messages.
3. **Train Data** — fits the baseline TF-IDF + Random Forest model and shows its metrics.
4. **Train Data GA** — runs genetic feature selection and reports the metrics for the reduced feature set.
5. **Spam Detection** — paste any message to classify it against the trained model.

## Notes

- The credentials are hard-coded for demo purposes. Replace them with Django's authentication framework before any real deployment.
- `views.py` uses `DataFrame.get_value()`, removed in pandas 1.0 — pin an older pandas or migrate to `.at[]` / `.iat[]`.
- Genetic feature selection is CPU-intensive; the first **Train Data GA** run takes noticeably longer than the baseline.
- `settings.py` ships with `DEBUG = True` and a development secret key. Change both before exposing the app.

## License

Released for academic and educational use.
