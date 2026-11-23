# Parkinson's Disease Prediction (ML)

Predicts whether a voice recording comes from a person with Parkinson's disease, using
biomedical voice measurements and a tuned Random Forest classifier. Includes a full Jupyter
workflow and a Streamlit app for interactive predictions.

> Educational project — not a medical diagnostic tool.

## Repository contents

```
.
├── Parkinson's Diseases.ipynb        # Full workflow: EDA, preprocessing, model comparison, tuning
├── Parkinsson disease_Dataset.csv    # Dataset: 195 voice recordings, 22 voice features + status label
└── Frontend/
    ├── app.py                        # Streamlit app (loads the trained model and scaler)
    ├── best_model.pkl                # Trained Random Forest
    ├── scaler.pkl                    # Feature scaler used in training
    └── requirements.txt
```

## Approach

1. **Data:** 195 recordings with features such as fundamental frequency (Fo, Fhi, Flo), jitter,
   shimmer, NHR/HNR, RPDE, DFA and PPE; `status` = 1 (Parkinson's) or 0 (healthy).
2. **Preprocessing:** cleaning, feature scaling, train/test split.
3. **Models compared:** SVM, Decision Tree, Random Forest, Gradient Boosting.
4. **Tuning:** grid search with 5-fold cross-validation on the Random Forest
   (best: 300 trees, max depth 7).

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | 0.881 | 0.911 | 0.932 | 0.921 |
| Decision Tree | 0.848 | 0.889 | 0.909 | 0.899 |
| **Random Forest** | **0.949** | **0.936** | **1.000** | **0.967** |
| Gradient Boosting | 0.898 | 0.932 | 0.932 | 0.932 |

Tuned Random Forest: **0.937 mean cross-validation accuracy**.

**Limitations:** the dataset is small, and several recordings come from the same person, so a
random split can place one person's recordings in both training and test sets. Real-world
accuracy would likely be lower; a person-level split is the next improvement.

## Run the app

```bash
cd Frontend
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py             # opens http://localhost:8501
```

To explore the workflow, open `Parkinson's Diseases.ipynb` in Jupyter or VS Code.

## Tech

Python · pandas · NumPy · scikit-learn · Matplotlib / Seaborn · Streamlit

## Author

**Naina Shukla** · [GitHub](https://github.com/Naina7120284) · [LinkedIn](https://www.linkedin.com/in/naina-shukla-15406a260/)
