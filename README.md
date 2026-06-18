# parkinson-prediction
Parkinson’s Disease Prediction Using ML This application predicts the likelihood of Parkinson’s disease using a trained ML model (Random Forest/ SVM). Features are extracted from voice measurements and processed using a scaler before prediction. The web interface is built with Streamlit for easy and interactive usage.
# Parkinson's Disease Prediction Project

##  **Project Overview**

Parkinson's Disease is a progressive neurological disorder affecting movement. Early detection can help manage symptoms better. This model uses machine learning algorithms to classify whether a patient is likely to have Parkinson's disease.

---

## **Features**

* Data preprocessing and cleaning
* Feature selection and visualization
* ML model training and testing
* Accuracy evaluation
* Streamlit web app interface

---

##  **Project Structure**

```
Parkinson's Prediction/
│── data/                 # Dataset CSV file
│── model/                # Trained model files
│── app.py                # Streamlit deployment file
│── notebook.ipynb        # Full ML workflow
│── requirements.txt      # List of dependencies
│── README.md             # Documentation
```

---

##  **Technologies Used**

* Python 3.x
* Pandas
* NumPy
* Scikit-Learn
* Streamlit (for UI)
* Matplotlib / Seaborn

---

##  **Installation Steps**

### 1️. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️. Run the Notebook (Optional)

Use Jupyter Notebook or VS Code.

---

##   **Running the Streamlit App**

```bash
streamlit run app.py
```

After running, Streamlit will generate a localhost URL. Open it in your browser.

---
Machine Learning Models Trained & Evaluated.
The following Machine Learning models were trained and evaluated:
.SVC
.Decision Tree Classifier
.Random Forest Classifier
.Gradient boosting classifier

##  **Model Performance**
Random Forest Classifier was found to be the best performing Classifier with:
 
===Best model is: Random Forest ===

.Accuracy: 0.9487

.Precision: 0.9412

.Recall: 1.0000

.F1 Score: 0.9697
```

```

4. Deploy on:
   
   * **Streamlit Cloud**
    web: streamlit run app.py
---

##   **Use Case**

This project can assist:

* Medical researchers
* Health practitioners
* Students learning ML in healthcare

---

##   **Author**

Developed by **Naina Shukla**.

---
